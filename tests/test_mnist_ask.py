"""`doomformer mnist` and `doomformer ask`: the shipped MNIST MLP, reading your own drawing, and the oracle's prompt."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

import doomformer as df
from doomformer import cli, intref, shipped


@pytest.fixture(scope="module")
def mnist():
    return df.MLP.load(shipped.data("mnist/mlp.npz"))


def drawn(tmp_path, name, ink=0, paper=255, strokes=(), ellipse=None):
    im = Image.new("L", (300, 400), paper)
    d = ImageDraw.Draw(im)
    for s in strokes:
        d.line(s, fill=ink, width=18, joint="curve")
    if ellipse:
        d.ellipse(ellipse, outline=ink, width=18)
    im.save(tmp_path / name)
    return tmp_path / name


def read(net, bits):
    """The digit of the reference one-hot (the shipped net ends in winner layers: DOOM's dolls pick the digit)."""
    return cli.doom_digit(net.reference(net.encode(bits.astype(np.float64)))["y"])


def test_shipped_model_reads_the_shipped_test_digits(mnist):
    d = np.load(shipped.data("mnist/digits.npz"))
    bits = np.unpackbits(d["bits"], axis=1)[:, :784]
    right = sum(read(mnist, b) == lab for b, lab in zip(bits, d["labels"]))
    assert len(bits) == 100 and right >= 98
    assert [W.shape for W in mnist.weights[1:]] == [(45, mnist.weights[0].shape[0]), (10, 45), (10, 10)]


def test_one_hot_reader_refuses_anything_else():
    assert cli.doom_digit([0, 0, 1, 0]) == 2
    for y in ([0, 0, 0], [1, 1, 0], [0, 2, 0]):
        with pytest.raises(ValueError):
            cli.doom_digit(y)


def test_drawings_are_prepared_like_mnist_and_read(tmp_path, mnist):
    seven = drawn(tmp_path, "7.png", strokes=[[(60, 60), (240, 60), (130, 340)]])
    one = drawn(tmp_path, "1.png", strokes=[[(120, 90), (160, 50), (160, 350)]])
    zero = drawn(tmp_path, "0.png", ellipse=(70, 50, 230, 350))
    for path, digit in ((seven, 7), (one, 1), (zero, 0)):
        bits = cli.digit_from_image(path)
        assert bits.shape == (784,) and set(np.unique(bits)) <= {0, 1}
        rows, cols = np.nonzero(bits.reshape(28, 28))
        assert max(rows.max() - rows.min(), cols.max() - cols.min()) <= 20          # scaled into MNIST's 20x20 box
        assert read(mnist, bits) == digit
    chalk = drawn(tmp_path, "7w.png", ink=255, paper=0, strokes=[[(60, 60), (240, 60), (130, 340)]])
    assert np.array_equal(cli.digit_from_image(chalk), cli.digit_from_image(seven))  # light on dark: same pixels


def test_blank_picture_is_refused(tmp_path):
    Image.new("L", (50, 50), 255).save(tmp_path / "blank.png")
    with pytest.raises(ValueError, match="no ink"):
        cli.digit_from_image(tmp_path / "blank.png")


@pytest.mark.timeout(1500)                  # the MNIST map (~160k sectors): 92 s here, 5+ min on a 3-core CI runner
def test_cli_mnist_reads_a_digit_in_doom_exactly(capsys):
    assert cli.main(["mnist", "--digits", "1", "--workers", "1", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "1/1 EXACT" in out and "read correctly" in out and "score comparisons" in out


def test_ask_prompt_keeps_only_what_the_oracle_can_read():
    vocab = df.Transformer.load("doomoracle").vocab
    assert cli.ask_prompt("What does IDDQD do?", vocab) == ("WHAT DOES IDDQD DO>", "")
    assert cli.ask_prompt("  what is   love?? <3", vocab) == ("WHAT IS LOVE>", "3<")
    assert cli.ask_prompt("???", vocab) == ("", "")


def test_oracle_answers_iddqd():
    oracle = df.Transformer.load("doomoracle")
    assert intref.generate(oracle, "WHAT DOES IDDQD DO>") == "WHAT DOES IDDQD DO>GOD MODE"
