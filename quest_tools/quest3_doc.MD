
# Quest 3 Setup & Connection Guide

Follow these steps to prepare and connect your Meta Quest 3 for development, controller data streaming, and debugging.

---

## 1. Initial Setup

- **Reference:** [Quest 3 Mobile Device Setup Guide](https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/)
- **Required:** Install the **Meta Quest Developer Hub** app on your computer.
- **App Installation:**
  Download and install the custom Quest APK from:
  [https://drive.google.com/drive/u/4/folders/11TVTuQ4d-HFZ9nxqva6ckv4SXSjxxTYw](https://drive.google.com/drive/u/4/folders/11TVTuQ4d-HFZ9nxqva6ckv4SXSjxxTYw)

---

## 2. Disable Boundary & Increase Sleep Timeout

### a) Enable Developer Mode

1. **On your smartphone:**
   - Install the Meta Quest app and connect it to your headset.
   - Go to **Devices → Developer Mode → Enable**.

2. **On Quest 3:**
   - Put on the headset.
   - Go to **Settings → System → Developer → Boundary** and turn it off.

---

## 3. Connect Quest 3 to Computer

There are two ways to connect Quest 3 for data transfer and debugging:

### **Method 1: Wi-Fi (Wireless, but may be unstable in China)**

1. Connect Quest 3 and your computer to the same Wi-Fi network.
2. Find Quest 3's IP in headset Wi-Fi settings.
   - Note: Wi-Fi can be slow/unstable; Quest may ping servers that are blocked in China.

### **Method 2: USB + ADB Port Forwarding (Recommended for reliability)**

#### A) **Connect via USB and Check Connection**

1. Connect Quest 3 to your computer with a USB data cable (not just charging).
2. Make sure **USB Debugging** is enabled:
   - On Quest 3: **Settings → Developer → Enable USB Debugging**.
   - When prompted, select **Allow** (and check "Always allow" if you prefer).

3. **Check USB Connection:**
   ```bash
   lsusb
   ```
   Example output:
   ```
   Bus 003 Device 004: ID 2833:0186 Meta Platforms, Inc. Quest 2/3/Pro
   ```
   - Vendor ID: `2833` (Meta)
   - Product ID: `0186`

#### B) **Troubleshooting if Quest 3 Doesn't Show Up**

- Double-check USB debugging is enabled on the Quest.
- Use a proper USB 3.0 cable (many cables only charge).
- Switch USB ports—prefer motherboard ports over hubs/docks.
- Ensure your Quest 3 developer account is active.

#### C) **If Still Not Detected (Linux): Set up udev Rules**

1. Edit rules:
   ```bash
   sudo nano /etc/udev/rules.d/51-android.rules
   ```
2. Add:
   ```
   SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"
   ```
3. Reload and restart:
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   adb kill-server
   adb start-server
   ```
4. **Check device:**
   ```bash
   adb devices
   ```
   - It should now display as **device**.

#### D) **ADB Port Forwarding**

Forward a port to allow your PC to read data from the Quest:
```bash
adb forward tcp:12345 tcp:12345
```
- Now you can connect to `127.0.0.1:12345` from your computer to read data.

---

## 4. Run the Custom APK on Quest 3

1. On Quest 3, open **Libraries** (bottom right), select **Unknown Sources** (left column).
2. Locate the app named **"Quest3 Reader"**.
3. Run it—you'll see an empty world as a test.

- **Tip:** If boundary security settings are still active, use **Meta Quest Developer Hub** to disable them.

---

## 6. Code Structure Overview

- **QuestReader**: Singleton class, reads head/hand positions (world coordinates), button states, etc. Only one instance is used, regardless of how many times you call it.
- **SingleArmQuest3Agent**: Uses QuestReader for transformations, updating the robot's end-effector pose based on Quest controller, and handles "clutch" button logic.

---

### **Summary Table: Key Steps**

| Step | Action                                                 |
|------|-------------------------------------------------------|
| 1    | Enable Developer Mode & USB Debugging                 |
| 2    | Confirm USB detection (`lsusb`)                       |
| 3    | Fix USB issues (change cable/port, udev rules, etc.)  |
| 4    | Forward ADB port: `adb forward tcp:12345 tcp:12345`   |
| 5    | Run the custom APK in Unknown Sources on Quest 3      |
| 6    | Run the Quest Agent Python code on your computer      |
