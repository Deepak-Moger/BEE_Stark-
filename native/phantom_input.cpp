// Optional compiled C ABI for deployments that require a native shared library.
// The Python ctypes backend remains the portable fallback and both expose the same
// safety boundary: this library only injects events when explicitly loaded.
#include <cstdint>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <ApplicationServices/ApplicationServices.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <linux/input.h>
#include <linux/uinput.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#endif

#if defined(_WIN32)
#define PHANTOM_EXPORT __declspec(dllexport)
#else
#define PHANTOM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
PHANTOM_EXPORT int phantom_move(int x, int y) {
#if defined(_WIN32)
    const int width = GetSystemMetrics(SM_CXSCREEN) - 1;
    const int height = GetSystemMetrics(SM_CYSCREEN) - 1;
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dx = static_cast<LONG>(static_cast<double>(x) * 65535.0 / (width > 0 ? width : 1));
    input.mi.dy = static_cast<LONG>(static_cast<double>(y) * 65535.0 / (height > 0 ? height : 1));
    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
    return SendInput(1, &input, sizeof(INPUT)) == 1 ? 0 : -1;
#elif defined(__APPLE__)
    CGEventRef event = CGEventCreateMouseEvent(nullptr, kCGEventMouseMoved, CGPointMake(x, y), kCGMouseButtonLeft);
    if (!event) return -1;
    CGEventPost(kCGHIDEventTap, event);
    CFRelease(event);
    return 0;
#elif defined(__linux__)
    static int fd = -1;
    static int old_x = 0, old_y = 0;
    if (fd < 0) {
        fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
        if (fd < 0) return -1;
        ioctl(fd, UI_SET_EVBIT, EV_REL); ioctl(fd, UI_SET_RELBIT, REL_X); ioctl(fd, UI_SET_RELBIT, REL_Y);
        ioctl(fd, UI_SET_EVBIT, EV_KEY); ioctl(fd, UI_SET_KEYBIT, BTN_LEFT); ioctl(fd, UI_SET_KEYBIT, BTN_RIGHT);
        uinput_user_dev device{}; std::strncpy(device.name, "PhantomOS Quantum", UINPUT_MAX_NAME_SIZE - 1);
        device.id.bustype = BUS_USB; device.id.vendor = 0x1234; device.id.product = 0x0001; device.id.version = 1;
        if (write(fd, &device, sizeof(device)) < 0 || ioctl(fd, UI_DEV_CREATE) < 0) return -1;
    }
    input_event events[3]{};
    events[0].type = EV_REL; events[0].code = REL_X; events[0].value = x - old_x;
    events[1].type = EV_REL; events[1].code = REL_Y; events[1].value = y - old_y;
    events[2].type = EV_SYN; events[2].code = SYN_REPORT;
    old_x = x; old_y = y;
    return write(fd, events, sizeof(events)) == static_cast<ssize_t>(sizeof(events)) ? 0 : -1;
#else
    (void)x; (void)y; return -1;
#endif
}

PHANTOM_EXPORT int phantom_button(int button, int pressed) {
#if defined(_WIN32)
    const DWORD down[] = {MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_MIDDLEDOWN};
    const DWORD up[] = {MOUSEEVENTF_LEFTUP, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_MIDDLEUP};
    INPUT input{}; input.type = INPUT_MOUSE; input.mi.dwFlags = (pressed ? down : up)[button < 0 || button > 2 ? 0 : button];
    return SendInput(1, &input, sizeof(INPUT)) == 1 ? 0 : -1;
#elif defined(__APPLE__)
    const CGEventType down[] = {kCGEventLeftMouseDown, kCGEventRightMouseDown, kCGEventOtherMouseDown};
    const CGEventType up[] = {kCGEventLeftMouseUp, kCGEventRightMouseUp, kCGEventOtherMouseUp};
    CGEventRef event = CGEventCreateMouseEvent(nullptr, (pressed ? down : up)[button < 0 || button > 2 ? 0 : button], CGPointZero, static_cast<CGMouseButton>(button));
    if (!event) return -1; CGEventPost(kCGHIDEventTap, event); CFRelease(event); return 0;
#elif defined(__linux__)
    (void)button; (void)pressed; return -1; // Device initialization is deployment-specific.
#else
    (void)button; (void)pressed; return -1;
#endif
}

PHANTOM_EXPORT int phantom_scroll(int dx, int dy) {
#if defined(_WIN32)
    INPUT input{}; input.type = INPUT_MOUSE; input.mi.mouseData = static_cast<DWORD>(dy * WHEEL_DELTA);
    input.mi.dwFlags = MOUSEEVENTF_WHEEL; return SendInput(1, &input, sizeof(INPUT)) == 1 ? 0 : -1;
#elif defined(__APPLE__)
    CGEventRef event = CGEventCreateScrollWheelEvent(nullptr, kCGScrollEventUnitLine, 2, dy, dx);
    if (!event) return -1; CGEventPost(kCGHIDEventTap, event); CFRelease(event); return 0;
#else
    (void)dx; (void)dy; return -1;
#endif
}

PHANTOM_EXPORT int phantom_key(int keycode, int pressed) {
#if defined(_WIN32)
    INPUT input{}; input.type = INPUT_KEYBOARD; input.ki.wVk = static_cast<WORD>(keycode);
    input.ki.dwFlags = pressed ? 0 : KEYEVENTF_KEYUP; return SendInput(1, &input, sizeof(INPUT)) == 1 ? 0 : -1;
#elif defined(__APPLE__)
    CGEventRef event = CGEventCreateKeyboardEvent(nullptr, static_cast<CGKeyCode>(keycode), pressed);
    if (!event) return -1; CGEventPost(kCGHIDEventTap, event); CFRelease(event); return 0;
#else
    (void)keycode; (void)pressed; return -1;
#endif
}
}
