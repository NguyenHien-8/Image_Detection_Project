# Image_Detection_Project
- Developer: Trần Nguyên Hiền
- Major: Electronics and Communication Engineering
- School: CAN THO UNIVERSITY
- Email: trannguyenhien29085@gmail.com
-----------------------------------------------------

```
anti-spoofing/
├── include/
│   ├── layer1_capture.h
│   ├── layer2_detection.h
│   ├── layer3_liveness.h
│   ├── layer4_hybrid.h (UPDATED)
│   └── anti_spoof_decision.h (NEW!)
├── src/
│   ├── main.cpp (REFACTORED)
│   ├── layer1_capture.cpp
│   ├── layer2_detection.cpp
│   ├── layer3_liveness.cpp (OPTIMIZED)
│   ├── layer4_hybrid.cpp (OPTIMIZED)
│   └── anti_spoof_decision.cpp (NEW!)
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── MiniFASNetV1SE.onnx
├── CMakeLists.txt (UPDATED)
└── BUILD_GUIDE.md (this file)
```

------------------------------------------------- Installation instructions --------------------------------------------------
```
git clone https://github.com/NguyenHien-8/Hardware-Library.git
```
### 1.Windows
**The command automatically creates directories for all branches**
- On PowerShell
```
git branch -r | % { $n = $_.Trim() -replace 'origin/'; if($n -notmatch 'HEAD'){ git worktree add $n $n } }
```
- On CMD
```
for /f "tokens=1,* delims=/" %a in ('git branch -r ^| findstr /v "HEAD"') do git worktree add "%b" "origin/%b"
```

### 2.Linux
**The command automatically creates directories for all branches.**
- On Bash
```
git branch -r | grep -v 'HEAD' | cut -d/ -f2- | xargs -I{} git worktree add {} origin/{}
```

### 3.MacOS
**The command automatically creates directories for all branches.**
- On Zsh/Bash
```
git branch -r | grep -v 'HEAD' | cut -d/ -f2- | xargs -I{} git worktree add {} origin/{}
```

Thank you,

Regards,
NguyenHien