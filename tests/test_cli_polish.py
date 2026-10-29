"""First-use behaviour of the command line that needs no DOOM: output paths, --version, argument checks, the picture
preparation of `mnist --image`, the missing-tool message and the `ask` time estimate."""
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import doomformer as df
from doomformer import acc, cli, shipped


def drawn(path, strokes, size=(300, 400), width=18):
    im = Image.new("L", size, 255)
    d = ImageDraw.Draw(im)
    for s in strokes:
        d.line(s, fill=0, width=width, joint="curve")
    im.save(path)
    return path


def text_page(path):
    """A 1200x1600 page of grey paper with 40 lines of small dark 'words'."""
    rng = np.random.default_rng(0)
    im = Image.new("L", (1200, 1600), 235)
    d = ImageDraw.Draw(im)
    for line in range(40):
        x = 80
        while x < 1000:
            w = int(rng.integers(30, 120))
            d.rectangle((x, 60 + line * 36, x + w, 60 + line * 36 + 14), fill=40)
            x += w + 24
    im.save(path)
    return path


def no_tools(monkeypatch, tmp_path):
    """PATH holds no git, cmake or compiler, and no acc is found or cached."""
    monkeypatch.setenv("PATH", str(tmp_path))
    for var in ("CC", "DOOMFORMER_ACC"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DOOMFORMER_ACC_IGNORE_REPO", "1")
    monkeypatch.setenv("DOOMFORMER_ACC_CACHE", str(tmp_path / "acc-cache"))


# --- picture preparation ---------------------------------------------------------------------------------------------
def test_clean_drawn_digits_and_the_shipped_digits_raise_no_warning(tmp_path):
    three = drawn(tmp_path / "3.png", [[(80, 70), (220, 70), (140, 180), (230, 250), (200, 330), (70, 330)]])
    seven = drawn(tmp_path / "7.png", [[(60, 60), (240, 60), (130, 340)]])
    tight = drawn(tmp_path / "7t.png", [[(9, 9), (291, 9), (150, 391)]])         # ink spans the whole picture
    for path in (three, seven, tight):
        assert cli.prepare_digit(path)[1] is None
    d = np.load(shipped.data("mnist/digits.npz"))
    for i, bits in enumerate(np.unpackbits(d["bits"], axis=1)[:, :784]):
        Image.fromarray(bits.reshape(28, 28) * np.uint8(255)).save(tmp_path / "s.png")
        assert cli.prepare_digit(tmp_path / "s.png")[1] is None, f"shipped test digit {i}"


def test_a_page_of_text_is_an_error_with_exit_2(tmp_path, capsys):
    from PIL import ImageFont
    pages = [text_page(tmp_path / "page.png")]
    im = Image.new("L", (1200, 1600), 255)
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(24)
    for line in range(40):
        d.text((80, 60 + line * 36), "the quick brown fox jumps over the lazy dog and DOOM", fill=0, font=font)
    im.save(tmp_path / "text.png")
    pages.append(tmp_path / "text.png")
    for page in pages:
        with pytest.raises(ValueError, match="nothing to read: after scaling to 28x28 no ink is left"):
            cli.prepare_digit(page)
        with pytest.raises(SystemExit) as e:
            cli.main(["mnist", "--image", str(page)])
        assert e.value.code == 2 and "nothing to read" in capsys.readouterr().err


def test_too_thin_too_fat_and_too_many_blobs_warn(tmp_path):
    thin = drawn(tmp_path / "thin.png", [[(60, 60), (240, 60), (130, 340)]], width=5)
    assert cli.DIGIT_WARNING in cli.prepare_digit(thin)[1] and "keeps 1 ink pixels" in cli.prepare_digit(thin)[1]
    im = Image.new("L", (300, 300), 255)
    ImageDraw.Draw(im).ellipse((50, 50, 250, 250), fill=0)
    im.save(tmp_path / "blob.png")
    assert "covers 40%" in cli.prepare_digit(tmp_path / "blob.png")[1]
    im = Image.new("L", (300, 300), 255)
    d = ImageDraw.Draw(im)
    for x, y in [(30, 30), (250, 40), (140, 150), (40, 250), (250, 250)]:
        d.ellipse((x - 25, y - 25, x + 25, y + 25), fill=0)
    im.save(tmp_path / "dots.png")
    assert "more than 3 separate blobs" in cli.prepare_digit(tmp_path / "dots.png")[1]


def test_blank_picture_is_an_error_with_exit_2(tmp_path, capsys):
    Image.new("L", (50, 50), 255).save(tmp_path / "blank.png")
    with pytest.raises(ValueError, match="no ink"):
        cli.prepare_digit(tmp_path / "blank.png")
    with pytest.raises(SystemExit) as e:
        cli.main(["mnist", "--image", str(tmp_path / "blank.png")])
    assert e.value.code == 2
    assert "no ink found" in capsys.readouterr().err


def test_image_preview_prints_before_the_run_and_the_warning_goes_to_stderr(tmp_path, capsys):
    args = types.SimpleNamespace(image=str(drawn(tmp_path / "page.png", [[(60, 60), (240, 60), (130, 340)]], width=5)))
    digits, labels, names, _ = cli._mnist_digits(args)
    out, err = capsys.readouterr()
    assert digits.shape == (1, 784) and names == ["page.png"] and digits.sum() == 1
    assert "as the 28x28 picture DOOM will read" in out and len(out.splitlines()) == 15
    assert err.startswith("doomformer mnist: warning: " + cli.DIGIT_WARNING)
    seven = drawn(tmp_path / "7.png", [[(60, 60), (240, 60), (130, 340)]])
    cli._mnist_digits(types.SimpleNamespace(image=str(seven)))
    assert capsys.readouterr().err == ""


def test_a_text_file_is_not_an_image(tmp_path, capsys):
    (tmp_path / "t.txt").write_text("hello")
    with pytest.raises(SystemExit) as e:
        cli.main(["mnist", "--image", str(tmp_path / "t.txt")])
    assert e.value.code == 2 and "is not an image" in capsys.readouterr().err


# --- output paths, version, arguments -------------------------------------------------------------------------------
def test_unwritable_outputs_stop_before_any_work(tmp_path, capsys):
    for argv in (["demo", "--png", str(tmp_path / "no" / "such" / "dir.png")],
                 ["mnist", "--record", str(tmp_path)]):                                  # a directory
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        err = capsys.readouterr().err
        assert e.value.code == 2 and f"doomformer {argv[0]}: cannot write {argv[2]}: " in err
        assert "Traceback" not in err


def test_the_path_check_leaves_no_file_and_makes_record_parents(tmp_path):
    cli._check_writable(str(tmp_path / "a.png"), "demo")
    assert not (tmp_path / "a.png").exists()
    cli._check_writable(str(tmp_path / "x" / "y" / "r.json"), "mnist", parents=True)
    assert (tmp_path / "x" / "y").is_dir() and not (tmp_path / "x" / "y" / "r.json").exists()
    (tmp_path / "old.png").write_bytes(b"keep")
    cli._check_writable(str(tmp_path / "old.png"), "demo")
    assert (tmp_path / "old.png").read_bytes() == b"keep"


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"doomformer {cli._version()}" and out.split()[1][0].isdigit()


def test_hello_rejects_zero_cases(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["hello", "--cases", "0"])
    assert e.value.code == 2 and "must be at least 1" in capsys.readouterr().err


def test_every_subcommand_has_help_text(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for name in ("demo", "mnist", "draw", "ask", "hello", "plan", "play", "doctor"):
        line = next(l for l in out.splitlines() if l.startswith(f"    {name} ") or l.startswith(f"    {name}\n"))
        assert len(line.split(None, 1)) == 2, name
    assert "--version" in out and "command" in out.split("positional arguments:")[1]


def test_plan_of_a_missing_model_file_says_so(tmp_path):
    with pytest.raises(SystemExit, match="no such file"):
        cli.main(["plan", str(tmp_path / "gone.npz")])


# --- missing build tools ---------------------------------------------------------------------------------------------
def test_missing_tools_are_named_with_install_hints_before_the_build_line(monkeypatch, tmp_path, capsys):
    no_tools(monkeypatch, tmp_path)
    with pytest.raises(acc.AccError) as e:
        cli._note_acc_build()
    msg = str(e.value)
    for hint in ("missing git (install: sudo apt install git", "cmake (install: sudo apt install cmake, or brew install "
                 "cmake)", "a C compiler (cc/gcc/clang) (install: sudo apt install build-essential, or xcode-select "
                 "--install on macOS)", "DOOMFORMER_ACC"):
        assert hint in msg
    assert "building the ACS compiler" not in capsys.readouterr().out
    with pytest.raises(acc.AccError, match="missing git"):
        acc.build(tmp_path / "src")


def test_doctor_reports_bad_and_exits_1_without_tools(monkeypatch, tmp_path, capsys):
    no_tools(monkeypatch, tmp_path)
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "BAD acc" in out and "cannot be built: missing git, cmake, a C compiler" in out


def test_main_reports_missing_tools_as_one_line(monkeypatch, tmp_path, capsys):
    no_tools(monkeypatch, tmp_path)
    assert cli.main(["hello"]) == 1
    cap = capsys.readouterr()
    assert cap.err.startswith("doomformer hello: ACC (the ACS compiler) is not available") and "Traceback" not in cap.err
    assert "building the ACS compiler" not in cap.out


def test_doctor_does_not_equate_tools_on_path_with_readiness(monkeypatch, capsys):
    from doomformer import udmf
    monkeypatch.setattr(acc, "find", lambda: (None, ""))
    monkeypatch.setattr(acc, "missing_tools", lambda: [])

    def failed_compile(source):
        raise acc.AccError("cmake build failed: a tool on PATH is not a working toolchain")

    monkeypatch.setattr(udmf, "compile_acs", failed_compile)
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "BAD acc" in out and "cmake build failed" in out and "all good" not in out


def test_acc_build_removes_upstream_static_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(acc, "require_tools", lambda: None)
    steps = []

    def run(step, cmd, **kwargs):
        steps.append(step)
        if step == "git clone":
            (Path(cmd[-1]) / "CMakeLists.txt").write_text(
                'set(FLAGS "${FLAGS} -static -static-libgcc -static-libstdc++")\n' * 3)
        elif step == "cmake configure":
            assert "-static" not in (Path(cmd[2]) / "CMakeLists.txt").read_text()
        elif step == "cmake build":
            binary = Path(cmd[2]) / "acc"
            binary.parent.mkdir()
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)

    monkeypatch.setattr(acc, "_run", run)
    assert acc.build(tmp_path / "src").is_file()
    assert steps == ["git clone", "git checkout", "cmake configure", "cmake build"]


# --- ask time estimate -----------------------------------------------------------------------------------------------
def test_ask_estimate_matches_the_logged_run_and_scales_with_workers():
    net = df.Transformer.load("doomoracle")
    assert sum(cli.ask_runs(net, t) for t in range(28)) == 5320             # the logged question: 28 steps, 5,320 runs
    total = lambda w: sum(cli.ask_estimate(net, 19, w)[:1]) + 8 * cli.ask_estimate(net, 19, w)[1]  # noqa: E731
    assert abs(total(6) - 653) < 0.1 * 653 and abs(total(3) - 1210) < 0.1 * 1210
    first = [cli.ask_estimate(net, 19, w)[0] for w in (1, 2, 4, 8)]
    assert first == sorted(first, reverse=True)
