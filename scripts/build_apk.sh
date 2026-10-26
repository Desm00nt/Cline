#!/usr/bin/env bash
# Builds a signed, installable debug APK without relying on Gradle.
# Requirements: JDK 17 and Android SDK (build-tools 34, platform android-33).
# Local defaults point at .tools/ used by the CI sandbox; override with env vars:
#   JAVA_HOME, ANDROID_SDK_ROOT, ANDROID_HOME
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$ROOT/.tools/sdk}}"
JAVA_HOME="${JAVA_HOME:-$ROOT/.tools/jdk}"
export PATH="$JAVA_HOME/bin:$PATH"
BT="$SDK/build-tools/34.0.0"
PLATFORM="$SDK/platforms/android-33/android.jar"
JAVA="$JAVA_HOME/bin/java"
JAVAC="$JAVA_HOME/bin/javac"
KEYTOOL="$JAVA_HOME/bin/keytool"
OUT="dist/RagdollArena-v1.0.apk"
B="$ROOT/build/apk"

rm -rf "$B"
mkdir -p "$B/res" "$B/gen" "$B/classes" "$B/dex" dist

echo "[1/6] compiling resources"
"$BT/aapt2" compile --dir app/src/main/res -o "$B/res.zip"

echo "[2/6] linking resources + manifest"
"$BT/aapt2" link -o "$B/app-unsigned.apk" \
    -I "$PLATFORM" \
    --manifest app/src/main/AndroidManifest.xml \
    -R "$B/res.zip" --auto-add-overlay \
    --java "$B/gen" \
    --min-sdk-version 21 --target-sdk-version 33 \
    --version-code 1 --version-name 1.0

echo "[3/6] compiling java"
find app/src/main/java "$B/gen" -name '*.java' > "$B/sources.txt"
"$JAVAC" -encoding UTF-8 --release 11 -classpath "$PLATFORM" \
    -d "$B/classes" @"$B/sources.txt"

echo "[4/6] dexing"
find "$B/classes" -name '*.class' > "$B/classes.txt"
"$BT/d8" --release --min-api 21 --lib "$PLATFORM" \
    --output "$B/dex" @"$B/classes.txt"
(cd "$B/dex" && zip -q "$B/app-unsigned.apk" classes.dex)

echo "[5/6] aligning"
"$BT/zipalign" -f 4 "$B/app-unsigned.apk" "$B/app-aligned.apk"

if [ ! -f tools/debug.keystore ]; then
    echo "creating debug keystore"
    "$KEYTOOL" -genkeypair -keystore tools/debug.keystore -storepass android \
        -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 \
        -validity 10000 -dname "CN=Android Debug,O=Android,C=US"
fi

echo "[6/6] signing"
"$BT/apksigner" sign --ks tools/debug.keystore --ks-pass pass:android \
    --key-pass pass:android --out "$OUT" "$B/app-aligned.apk"
"$BT/apksigner" verify "$OUT"

echo "APK ready: $OUT"
ls -lh "$OUT"
