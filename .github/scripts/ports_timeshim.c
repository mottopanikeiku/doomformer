/* LD_PRELOAD shim for ports.yml: runs an engine's clock TIMESHIM_SPEED times faster than real time, so a ZDoom-family port
 * that waits for 35 tics per second (ZDoom 2.8.1, Zandronum, LZDoom have no time-scale option) plays a 235,567-tic map in
 * minutes. It scales clock_gettime, gettimeofday, time and the sleeps of the calling program; game tics, ACS Delay and
 * everything the map computes count tics and are not changed.
 *   gcc -shared -fPIC -O2 -o ports_timeshim.so ports_timeshim.c -ldl
 *   env LD_PRELOAD=$PWD/ports_timeshim.so TIMESHIM_SPEED=50 gzdoom ...      (not behind stdbuf: stdbuf replaces LD_PRELOAD) */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/* stdbuf would replace LD_PRELOAD: make the program's own stdout line-buffered here instead */
__attribute__((constructor)) static void line_buffered(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
}

static double speed = 0.0;
static struct timespec base_mono, base_real;
static int (*real_clock_gettime)(clockid_t, struct timespec *);
static int (*real_gettimeofday)(struct timeval *, void *);
static int (*real_nanosleep)(const struct timespec *, struct timespec *);
static int (*real_clock_nanosleep)(clockid_t, int, const struct timespec *, struct timespec *);
static int (*real_usleep)(useconds_t);

static void init(void) {
    if (speed != 0.0) return;
    const char *s = getenv("TIMESHIM_SPEED");
    double v = s ? atof(s) : 1.0;
    real_clock_gettime = dlsym(RTLD_NEXT, "clock_gettime");
    real_gettimeofday = dlsym(RTLD_NEXT, "gettimeofday");
    real_nanosleep = dlsym(RTLD_NEXT, "nanosleep");
    real_clock_nanosleep = dlsym(RTLD_NEXT, "clock_nanosleep");
    real_usleep = dlsym(RTLD_NEXT, "usleep");
    real_clock_gettime(CLOCK_MONOTONIC, &base_mono);
    real_clock_gettime(CLOCK_REALTIME, &base_real);
    speed = v > 0 ? v : 1.0;
}

static void scale(struct timespec *t, const struct timespec *base) {
    double d = (double)(t->tv_sec - base->tv_sec) + (double)(t->tv_nsec - base->tv_nsec) * 1e-9;
    d *= speed;
    double s = (double)base->tv_sec + (double)base->tv_nsec * 1e-9 + d;
    t->tv_sec = (time_t)s;
    t->tv_nsec = (long)((s - (double)t->tv_sec) * 1e9);
}

int clock_gettime(clockid_t c, struct timespec *t) {
    init();
    int r = real_clock_gettime(c, t);
    if (r == 0) {
        if (c == CLOCK_MONOTONIC || c == CLOCK_MONOTONIC_RAW || c == CLOCK_BOOTTIME || c == CLOCK_MONOTONIC_COARSE) scale(t, &base_mono);
        else if (c == CLOCK_REALTIME || c == CLOCK_REALTIME_COARSE) scale(t, &base_real);
    }
    return r;
}

int gettimeofday(struct timeval *tv, void *tz) {
    init();
    int r = real_gettimeofday(tv, tz);
    if (r == 0 && tv) {
        struct timespec t = {tv->tv_sec, tv->tv_usec * 1000};
        scale(&t, &base_real);
        tv->tv_sec = t.tv_sec;
        tv->tv_usec = t.tv_nsec / 1000;
    }
    return r;
}

time_t time(time_t *out) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    if (out) *out = tv.tv_sec;
    return tv.tv_sec;
}

static struct timespec shrink(const struct timespec *req) {
    double d = ((double)req->tv_sec + (double)req->tv_nsec * 1e-9) / speed;
    struct timespec r = {(time_t)d, (long)((d - (double)(time_t)d) * 1e9)};
    return r;
}

int nanosleep(const struct timespec *req, struct timespec *rem) {
    init();
    struct timespec r = shrink(req);
    return real_nanosleep(&r, rem);
}

int clock_nanosleep(clockid_t c, int flags, const struct timespec *req, struct timespec *rem) {
    init();
    if (flags & TIMER_ABSTIME) {            /* the deadline is in the program's (scaled) time: convert it back */
        const struct timespec *b = (c == CLOCK_REALTIME) ? &base_real : &base_mono;
        double d = ((double)(req->tv_sec - b->tv_sec) + (double)(req->tv_nsec - b->tv_nsec) * 1e-9) / speed;
        double t = (double)b->tv_sec + (double)b->tv_nsec * 1e-9 + d;
        struct timespec a = {(time_t)t, (long)((t - (double)(time_t)t) * 1e9)};
        return real_clock_nanosleep(c, flags, &a, rem);
    }
    struct timespec r = shrink(req);
    return real_clock_nanosleep(c, flags, &r, rem);
}

int usleep(useconds_t us) {
    init();
    return real_usleep((useconds_t)(us / speed));
}
