#!/bin/zsh
# Builds Soundxx.app: a tiny bundle that starts the app from this folder's venv.
# It carries a microphone usage description, so macOS asks "Soundxx would like
# to access the microphone" the first time. Ad-hoc signed so the permission sticks.
#   scripts/build_app.sh [destination folder, default: this repo]
set -e
REPO="${0:A:h:h}"
DEST="${1:-$REPO}"
APP="$DEST/Soundxx.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Soundxx</string>
  <key>CFBundleDisplayName</key><string>Soundxx</string>
  <key>CFBundleIdentifier</key><string>local.soundxx.app</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Soundxx</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>Soundxx records the lecture from the MacBook microphone and plays it, cleaned up, into your earbuds. The recording stays on this Mac.</string>
</dict>
</plist>
PLIST
cat > "$APP/Contents/MacOS/Soundxx" <<LAUNCH
#!/bin/zsh
cd "$REPO" || exit 1
exec "$REPO/venv/bin/python" -m focus_ear.app "\$@" >> "\$HOME/.focus-ear/app-stdout.log" 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/Soundxx"
codesign --force --deep -s - "$APP" 2>/dev/null || echo "(couldn't sign it ad hoc; it still runs)"
echo "Built $APP"
