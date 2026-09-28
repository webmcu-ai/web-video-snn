// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// VISION ML - ANN + SNN INFERENCE ONLY - v004 (real video, config-driven)
//
// Companion firmware to the browser trainer (index-v004.html). All training
// happens in the browser; this firmware COLLECTS video clips and RUNS
// INFERENCE ONLY - it never trains, exactly as in v002.
//
// WHAT CHANGED FROM THIS FILE'S v002 - TWO THINGS
//
// 1) CONFIG IS NOW READ FROM THE SD CARD, NOT COMPILED IN.
//    v002 hardcoded NUM_CLASSES, myClassLabels[], INPUT_SIZE, NUM_CHANNELS,
//    SNN_TIMESTEPS, and the LIF params as #defines/globals you had to edit
//    and reflash by hand. This version reads all of it from
//    /header/config.json on boot - the exact file index-v004.html writes
//    (Export All (.zip), or point "Choose Real Folder" at the SD card) - and
//    sizes every buffer at runtime to match. Add/rename/reorder classes,
//    change resolution, retrain - just update config.json on the card and
//    reboot. No #define to touch, no reflash. Expected shape:
//      { "classes": ["0Blank","1Cup","2Pen"], "inputSize": 64,
//        "numChannels": 3, "snnTimesteps": 16, "lifLeak": 0.9,
//        "lifThreshold": 0.5, "snnTraceLeak": 0.95,
//        "clipFrameIntervalMs": 80 }
//    If config.json is missing, the built-in default (classes 1Unknown,
//    2MovingHand, 3MovingFace, 64x64 RGB, 16 timesteps, leak 0.90,
//    threshold 0.50, trace leak 0.95, 80ms) is used AND written to
//    /header/config.json on the SD card, so there's a real file to edit.
//    Weight loads still need matching .bin files from the browser.
//
// 2) COLLECTS AND CLASSIFIES REAL VIDEO CLIPS, NOT SINGLE STILLS.
//    Collection (menu items 1..N) now records a whole CLIP per tap -
//    SNN_TIMESTEPS real frames, CLIP_FRAME_INTERVAL_MS apart - into
//    images/<folder>/clip_<id>/f00.jpg.. on the SD card, the same layout
//    index-v004.html's browser-side Store writes, so clips collected here
//    or in the browser mix freely.
//    SNN inference keeps a rolling window of the last SNN_TIMESTEPS real
//    camera frames and runs the FULL window through mySnnForwardInfer()
//    every tick - THERE IS NO EARLY EXIT (v002's SNN_MIN_TIMESTEPS /
//    SNN_COMMIT_MIN_PROB / SNN_COMMIT_MARGIN are gone). v002's early-exit
//    was correct for a still image: every synthetic timestep just re-rolled
//    noise from the SAME photo, so once the trace was confident, more steps
//    bought nothing worth waiting for. Real video is the opposite case -
//    every timestep IS new evidence (actual motion) - so exiting early here
//    would mean throwing away the newest real frames instead of a
//    redundant re-sample. Always run the whole window.
//    ANN inference is unchanged: it classifies whatever the single latest
//    frame is, every tick.
//    IMPORTANT: a mySnnWeights.bin trained on v002-style still-image clips
//    and one trained on v004-style real-video clips are byte-identical in
//    SIZE but not interchangeable in MEANING - the network learned
//    different assumptions about what changes between timesteps. Don't mix
//    them.
//
// SD card stores: images/<folder>/clip_<id>/f00.jpg.. (this firmware or the
//                 browser can write these; loose <folder>/img_*.jpg from
//                 v002 still count as 1-frame clips)
// SD card reads:  /header/config.json (classes + all settings)
//                 /header/myWeights.bin (ANN), /header/mySnnWeights.bin (SNN)
// Serial monitor and OLED output
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique), Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
// LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// For platformio you need the U8g2 library declared in the platformio.ini file and OPI PSRAM set
//   lib_deps = olikraus/U8g2 @ ^2.35.30
//   build_flags = -DBOARD_HAS_PSRAM -DARDUINO_USB_CDC_ON_BOOT=1
//   board_build.arduino.memory_type = qio_opi
//   board_build.flash_mode = qio
//   board_upload.flash_size = 8MB
//
// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Pins, Globals, Memory, Weights, Setup, Loop           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// optional Uncomment AFTER copying myWeights.h / mySnnWeights.h from SD to your sketch folder:
// Priority order: SD weights > baked-in weights > random He-init.
// NOTE: since architecture sizes are now runtime values (read from
// config.json) rather than #defines, a baked header generated for a
// DIFFERENT config.json (different class count / INPUT_SIZE / NUM_CHANNELS)
// would silently overrun these buffers. Regenerate your baked headers
// against the current config.json before enabling this, and see the size
// guard in setup() below.
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS
#ifdef USE_BAKED_WEIGHTS
  #include "myWeights.h"      // must also define BAKED_NUM_CLASSES, BAKED_INPUT_SIZE, BAKED_NUM_CHANNELS
  #include "mySnnWeights.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>
#include <Wire.h>

U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// CONFIGURATION - now read from /header/config.json at boot (myLoadConfig()
// below), not hardcoded. These globals hold whatever it (or the fallback)
// says. See the header comment above for the expected JSON shape.
// ======================================================
#define CONFIG_PATH "/header/config.json"
int NUM_CLASSES = 0;
std::vector<String> myClassLabels;   // display labels, straight from config.json
std::vector<String> myClassFolders;  // SD folder names (sanitized, matches the browser's rule)
int myTotalItems = 0;                // NUM_CLASSES + Infer ANN + Infer SNN

int INPUT_SIZE = 64;
int NUM_CHANNELS = 3;
int SNN_TIMESTEPS = 16;
float LIF_LEAK = 0.90f;
float LIF_THRESHOLD = 0.50f;
float SNN_TRACE_LEAK = 0.95f;
int CLIP_FRAME_INTERVAL_MS = 80;

const int myThresholdPress = 1100;
const int myThresholdRelease = 900;

// ======================================================
// UNIFIED TOUCH INPUT SYSTEM (unchanged from v44/v002)
// ======================================================
struct TouchState {
  bool isTouching = false;
  int tapCount = 0;
  unsigned long firstTapTime = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime = 0;
  const unsigned long tapWindow = 800;
  const int longPressTaps = 3;
  const unsigned long debounceDelay = 50;
};
TouchState myTouch;

unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime = 0;
const int myTapCooldown = 250;
int myMenuIndex = 1;
bool myIsSelected = false;
bool myAnnTrained = false;   // true once myWeights.bin loaded successfully
bool mySnnTrained = false;   // true once mySnnWeights.bin loaded successfully

// XIAO ESP32-S3 Camera Pins
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13
#define CAPTURE_SIZE 240   // camera always captures square 240x240 JPEGs regardless of INPUT_SIZE

// ======================================================
// CNN ARCHITECTURE CONSTANTS - shared by BOTH the ANN and the SNN.
// Kernel sizes / filter counts are fixed (the device doesn't make these
// configurable, matching index-v004.html); everything derived from them
// and from INPUT_SIZE/NUM_CHANNELS/NUM_CLASSES is now computed at runtime
// in myComputeArchitecture() once config.json has been read, since those
// three are no longer compile-time constants.
// ======================================================
const int CONV1_KERNEL_SIZE = 3;
const int CONV1_FILTERS = 4;
const int CONV2_KERNEL_SIZE = 3;
const int CONV2_FILTERS = 8;

int CONV1_W_PER_FILTER = 0, CONV1_WEIGHTS = 0;
int CONV2_W_PER_FILTER = 0, CONV2_WEIGHTS = 0;
int CONV1_OUTPUT_SIZE = 0, POOL1_OUTPUT_SIZE = 0, CONV2_OUTPUT_SIZE = 0;
int FLATTENED_SIZE = 0, OUTPUT_WEIGHTS = 0;
int INPUT_N = 0; // INPUT_SIZE*INPUT_SIZE*NUM_CHANNELS

void myComputeArchitecture() {
  CONV1_W_PER_FILTER = CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * NUM_CHANNELS;
  CONV1_WEIGHTS = CONV1_W_PER_FILTER * CONV1_FILTERS;
  CONV2_W_PER_FILTER = CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE * CONV1_FILTERS;
  CONV2_WEIGHTS = CONV2_W_PER_FILTER * CONV2_FILTERS;
  CONV1_OUTPUT_SIZE = INPUT_SIZE - (CONV1_KERNEL_SIZE - 1);
  POOL1_OUTPUT_SIZE = CONV1_OUTPUT_SIZE / 2;
  CONV2_OUTPUT_SIZE = POOL1_OUTPUT_SIZE - (CONV2_KERNEL_SIZE - 1);
  FLATTENED_SIZE = CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS;
  OUTPUT_WEIGHTS = FLATTENED_SIZE * NUM_CLASSES;
  INPUT_N = INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS;
}

// ======================================================
// GLOBAL VARIABLE DEFINITIONS
// ======================================================
uint8_t* myRgbBuffer = nullptr;     // reusable 240x240x3 RGB buffer (fixed - camera resolution, not INPUT_SIZE)
bool mySDavailable = false;
int* myResizeLookup = nullptr;      // [INPUT_SIZE] -> source pixel in the 240x240 frame (same lookup for x and y, square)

// ---- ANN weights (PSRAM) ----
float* myInputBuffer = nullptr;     // normalized 0-1 pixels, NUM_CHANNELS per pixel - the single latest frame
float* myConv1_w = nullptr;  float* myConv1_b = nullptr;
float* myConv2_w = nullptr;  float* myConv2_b = nullptr;
float* myOutput_w = nullptr; float* myOutput_b = nullptr;
float* myConv1_output = nullptr;
float* myPool1_output = nullptr;
float* myConv2_output = nullptr;
float* myDense_output = nullptr;    // ANN softmax probabilities [NUM_CLASSES]

// ---- SNN weights (PSRAM) - same shapes as the ANN's, different math ----
float* mySnnConv1_w = nullptr;  float* mySnnConv1_b = nullptr;
float* mySnnConv2_w = nullptr;  float* mySnnConv2_b = nullptr;
float* mySnnOutput_w = nullptr; float* mySnnOutput_b = nullptr;
float* mySnnDense_output = nullptr; // SNN softmax probabilities [NUM_CLASSES], after the full window

// ---- SNN streaming/scratch buffers (persist membrane potential across timesteps) ----
float*   mySnnC1Mem = nullptr;
float*   mySnnC2Mem = nullptr;
float*   mySnnTrace = nullptr;      // [NUM_CLASSES]
uint8_t* mySnnInputSpike = nullptr; // [INPUT_N] - this timestep's rate-coded frame
uint8_t* mySnnC1Spike = nullptr;
uint8_t* mySnnPooled = nullptr;
uint8_t* mySnnC2Spike = nullptr;    // [FLATTENED_SIZE]

// ---- Rolling window of real camera frames for video SNN inference ----
float** mySnnWindow = nullptr;      // [SNN_TIMESTEPS] pointers, each [INPUT_N] floats - a circular buffer
int mySnnWindowFilled = 0, mySnnWindowNext = 0;
bool mySnnVerbose = true;           // true: print every timestep + timing; false: final result only. Toggle with 'v' in Infer SNN.
float* mySnnStepProbs = nullptr;    // [NUM_CLASSES] scratch for the per-timestep running softmax (verbose mode)

// ======================================================
// UTILITY FUNCTIONS
// ======================================================
inline float clip_value(float v, float mn=-100, float mx=100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}
inline float leaky_relu(float x) { return x > 0 ? x : 0.1f * x; }

// ======================================================
// MINIMAL config.json READER - tailored to the exact shape index-v004.html
// writes (see the header comment), not a general-purpose JSON parser.
// ======================================================
int myJsonFindColon(const String &json, const String &key) {
  int i = json.indexOf("\"" + key + "\"");
  if (i < 0) return -1;
  return json.indexOf(':', i + key.length() + 2);
}
bool myJsonGetFloat(const String &json, const String &key, float &out) {
  int i = myJsonFindColon(json, key);
  if (i < 0) return false;
  i++;
  while (i < (int)json.length() && isspace(json[i])) i++;
  int start = i;
  while (i < (int)json.length() && (isDigit(json[i]) || json[i]=='-' || json[i]=='+' || json[i]=='.' || json[i]=='e' || json[i]=='E')) i++;
  if (i == start) return false;
  out = json.substring(start, i).toFloat();
  return true;
}
bool myJsonGetInt(const String &json, const String &key, int &out) {
  float f;
  if (!myJsonGetFloat(json, key, f)) return false;
  out = (int)lroundf(f);
  return true;
}
std::vector<String> myJsonGetStringArray(const String &json, const String &key) {
  std::vector<String> out;
  int i = myJsonFindColon(json, key);
  if (i < 0) return out;
  int arrStart = json.indexOf('[', i);
  int arrEnd = json.indexOf(']', arrStart < 0 ? i : arrStart);
  if (arrStart < 0 || arrEnd < 0) return out;
  int p = arrStart + 1;
  while (p < arrEnd) {
    int q1 = json.indexOf('"', p);
    if (q1 < 0 || q1 > arrEnd) break;
    int q2 = json.indexOf('"', q1 + 1);
    if (q2 < 0 || q2 > arrEnd) break;
    out.push_back(json.substring(q1 + 1, q2));
    p = q2 + 1;
  }
  return out;
}
// Same folder-sanitizing rule as sanitizeFolderName() in index-v004.html,
// so images/<folder>/ matches whatever the browser wrote for the same label.
String mySanitizeFolderName(const String &label) {
  String stripped;
  for (size_t i = 0; i < label.length(); i++) {
    char c = label[i];
    if (c=='/'||c=='\\'||c==':'||c=='*'||c=='?'||c=='"'||c=='<'||c=='>'||c=='|') continue;
    stripped += c;
  }
  stripped.trim();
  String out; bool lastWasSpace = false;
  for (size_t i = 0; i < stripped.length(); i++) {
    char c = stripped[i];
    if (isspace(c)) { if (!lastWasSpace) out += '_'; lastWasSpace = true; }
    else { out += c; lastWasSpace = false; }
  }
  if (out.length() == 0) out = "class";
  return out;
}
void myApplyFallbackConfig() {
  myClassLabels = { "1Unknown", "2MovingHand", "3MovingFace" };
  myClassFolders.clear();
  for (auto &l : myClassLabels) myClassFolders.push_back(mySanitizeFolderName(l));
  NUM_CLASSES = myClassLabels.size(); myTotalItems = NUM_CLASSES + 2;
  INPUT_SIZE = 64; NUM_CHANNELS = 3;
  SNN_TIMESTEPS = 16; LIF_LEAK = 0.90f; LIF_THRESHOLD = 0.50f; SNN_TRACE_LEAK = 0.95f;
  CLIP_FRAME_INTERVAL_MS = 80;
  Serial.println("No " CONFIG_PATH " found - using the built-in default config (1Unknown, 2MovingHand, 3MovingFace).");
}
// Writes the default config to the SD card as a real config.json the first
// time the device boots without one, so there's a file on the card to look
// at, edit, or replace with the browser's export. Same shape the browser
// writes. Does nothing if a config.json already exists.
void myWriteDefaultConfigIfMissing() {
  if (!mySDavailable || SD.exists(CONFIG_PATH)) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open(CONFIG_PATH, FILE_WRITE);
  if (!f) { Serial.println("Could not write default " CONFIG_PATH); return; }
  f.print("{\n  \"classes\": [");
  for (int i = 0; i < NUM_CLASSES; i++) { f.print(i ? ", " : ""); f.print("\"" + myClassLabels[i] + "\""); }
  f.printf("],\n  \"inputSize\": %d,\n  \"numChannels\": %d,\n  \"snnTimesteps\": %d,\n", INPUT_SIZE, NUM_CHANNELS, SNN_TIMESTEPS);
  f.printf("  \"lifLeak\": %.2f,\n  \"lifThreshold\": %.2f,\n  \"snnTraceLeak\": %.2f,\n  \"clipFrameIntervalMs\": %d\n}\n",
           LIF_LEAK, LIF_THRESHOLD, SNN_TRACE_LEAK, CLIP_FRAME_INTERVAL_MS);
  f.close();
  Serial.println("Wrote default " CONFIG_PATH " to the SD card - edit it or replace it with the browser's export.");
}
bool myLoadConfig() {
  if (!mySDavailable || !SD.exists(CONFIG_PATH)) return false;
  File f = SD.open(CONFIG_PATH, FILE_READ);
  if (!f) return false;
  String json = f.readString();
  f.close();

  std::vector<String> labels = myJsonGetStringArray(json, "classes");
  if (labels.empty()) { Serial.println(CONFIG_PATH " has no classes[] - ignoring it."); return false; }
  myClassLabels = labels;
  myClassFolders.clear();
  for (auto &l : myClassLabels) myClassFolders.push_back(mySanitizeFolderName(l));
  NUM_CLASSES = myClassLabels.size();
  myTotalItems = NUM_CLASSES + 2;

  int iv;
  if (myJsonGetInt(json, "inputSize", iv)) INPUT_SIZE = iv;
  if (myJsonGetInt(json, "numChannels", iv)) NUM_CHANNELS = iv;
  if (myJsonGetInt(json, "snnTimesteps", iv)) SNN_TIMESTEPS = iv;
  if (myJsonGetInt(json, "clipFrameIntervalMs", iv)) CLIP_FRAME_INTERVAL_MS = iv;
  float fv;
  if (myJsonGetFloat(json, "lifLeak", fv)) LIF_LEAK = fv;
  if (myJsonGetFloat(json, "lifThreshold", fv)) LIF_THRESHOLD = fv;
  if (myJsonGetFloat(json, "snnTraceLeak", fv)) SNN_TRACE_LEAK = fv;

  Serial.printf("[config] %d classes, %dx%dx%d, SNN_TIMESTEPS=%d, clip interval=%dms\n",
                NUM_CLASSES, INPUT_SIZE, INPUT_SIZE, NUM_CHANNELS, SNN_TIMESTEPS, CLIP_FRAME_INTERVAL_MS);
  for (int i = 0; i < NUM_CLASSES; i++) Serial.printf("  class %d: \"%s\" -> /images/%s\n", i, myClassLabels[i].c_str(), myClassFolders[i].c_str());
  return true;
}

// ======================================================
// UNIFIED TOUCH INPUT FUNCTIONS (unchanged from v44/v002)
// ======================================================
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}
void myResetTouchState() {
  myTouch.isTouching = false; myTouch.tapCount = 0;
  myTouch.firstTapTime = 0; myTouch.lastReleaseTime = 0; myTouch.lastCheckTime = 0;
}
void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;
  int val = myReadTouch();
  bool touchActive = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);
  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount = 1; myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }
  if (!touchActive && myTouch.isTouching) { myTouch.isTouching = false; myTouch.lastReleaseTime = now; }
}
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
      int count = myTouch.tapCount;
      myResetTouchState();
      if (result == 2) Serial.printf("LONG PRESS detected (%d taps)\n", count);
      else Serial.printf("TAP detected (%d tap%s)\n", count, count > 1 ? "s" : "");
      return result;
    }
  }
  return 0;
}
void myCheckTouchBackground() { myUpdateTouchState(); }

// ======================================================
// MEMORY ALLOCATION - sizes now come from the runtime architecture values
// computed by myComputeArchitecture(), which must be called first.
// ======================================================
void myBuildResizeLookup() {
  if (myResizeLookup) free(myResizeLookup);
  myResizeLookup = (int*)malloc(INPUT_SIZE * sizeof(int));
  for (int i = 0; i < INPUT_SIZE; i++) myResizeLookup[i] = min((int)((i + 0.5) * (float)CAPTURE_SIZE / INPUT_SIZE), CAPTURE_SIZE - 1);
}
void myAllocSnnWindow() {
  mySnnWindow = (float**)malloc(SNN_TIMESTEPS * sizeof(float*));
  for (int t = 0; t < SNN_TIMESTEPS; t++) mySnnWindow[t] = (float*)ps_malloc(INPUT_N * sizeof(float));
  mySnnWindowFilled = 0; mySnnWindowNext = 0;
}
void myAllocateMemory() {
  Serial.println("\n=== Allocating Memory ===");

  myInputBuffer = (float*)ps_malloc(INPUT_N * sizeof(float));

  myConv1_w  = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  myConv1_b  = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  myConv2_w  = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  myConv2_b  = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  myOutput_w = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  myConv1_output = (float*)ps_malloc(CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myPool1_output = (float*)ps_malloc(POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myConv2_output = (float*)ps_malloc(CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS*sizeof(float));
  myDense_output = (float*)ps_malloc(NUM_CLASSES*sizeof(float));

  mySnnConv1_w  = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  mySnnConv1_b  = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  mySnnConv2_w  = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  mySnnConv2_b  = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  mySnnOutput_w = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  mySnnOutput_b = (float*)ps_malloc(NUM_CLASSES * sizeof(float));
  mySnnDense_output = (float*)ps_malloc(NUM_CLASSES*sizeof(float));
  mySnnStepProbs = (float*)ps_malloc(NUM_CLASSES*sizeof(float));

  mySnnC1Mem      = (float*)ps_calloc(CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, sizeof(float));
  mySnnC2Mem      = (float*)ps_calloc(CONV2_FILTERS*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE, sizeof(float));
  mySnnTrace      = (float*)ps_calloc(NUM_CLASSES, sizeof(float));
  mySnnInputSpike = (uint8_t*)ps_malloc(INPUT_N*sizeof(uint8_t));
  mySnnC1Spike    = (uint8_t*)ps_malloc(CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*sizeof(uint8_t));
  mySnnPooled     = (uint8_t*)ps_malloc(CONV1_FILTERS*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*sizeof(uint8_t));
  mySnnC2Spike    = (uint8_t*)ps_malloc(FLATTENED_SIZE*sizeof(uint8_t));

  if (!myInputBuffer || !myConv1_w || !myConv2_w || !myOutput_w ||
      !myConv1_output || !myPool1_output || !myConv2_output ||
      !mySnnConv1_w || !mySnnConv2_w || !mySnnOutput_w || !mySnnC1Mem || !mySnnC2Mem) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage(); do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while (1) delay(1000);
  }
  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  // He-init both models with random weights (overwritten by SD/baked weights if present)
  float c1std = sqrt(2.0 / CONV1_W_PER_FILTER);
  for (int i=0;i<CONV1_WEIGHTS;i++){ myConv1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c1std; mySnnConv1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c1std; }
  for (int i=0;i<CONV1_FILTERS;i++){ myConv1_b[i]=0; mySnnConv1_b[i]=0; }
  float c2std = sqrt(2.0 / CONV2_W_PER_FILTER);
  for (int i=0;i<CONV2_WEIGHTS;i++){ myConv2_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c2std; mySnnConv2_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c2std; }
  for (int i=0;i<CONV2_FILTERS;i++){ myConv2_b[i]=0; mySnnConv2_b[i]=0; }
  float dstd = sqrt(2.0 / max(FLATTENED_SIZE,1));
  for (int i=0;i<OUTPUT_WEIGHTS;i++){ myOutput_w[i]=((float)rand()/RAND_MAX-0.5f)*2*dstd; mySnnOutput_w[i]=((float)rand()/RAND_MAX-0.5f)*2*dstd; }
  for (int i=0;i<NUM_CLASSES;i++){ myOutput_b[i]=0; mySnnOutput_b[i]=0; }
  Serial.println("He-init random weights set for ANN and SNN");
}

// ======================================================
// WEIGHT LOADING (read-only - training happens in the browser now).
// Same field order as modelSerialize() in index-v004.html; the size check
// below is what catches a config.json/weights-file mismatch (e.g. a
// mySnnWeights.bin trained for a different class count or resolution).
// ======================================================
bool myLoadWeightsFile(const char* path, float* c1w, float* c1b, float* c2w, float* c2b, float* ow, float* ob, const char* tag) {
  if (!mySDavailable) { Serial.printf("No SD card - skipping %s weight load\n", tag); return false; }
  if (!SD.exists(path)) { Serial.printf("No %s found\n", path); return false; }
  File f = SD.open(path, FILE_READ);
  if (!f) return false;
  size_t totalFloats = (size_t)CONV1_WEIGHTS + CONV1_FILTERS + CONV2_WEIGHTS + CONV2_FILTERS + OUTPUT_WEIGHTS + NUM_CLASSES;
  if (f.size() != totalFloats * 4) {
    Serial.printf("%s size %u != expected %u for the current config.json - architecture mismatch, NOT loaded.\n",
                  path, (unsigned)f.size(), (unsigned)(totalFloats * 4));
    f.close(); return false;
  }
  f.read((uint8_t*)c1w, CONV1_WEIGHTS*4);
  f.read((uint8_t*)c1b, CONV1_FILTERS*4);
  f.read((uint8_t*)c2w, CONV2_WEIGHTS*4);
  f.read((uint8_t*)c2b, CONV2_FILTERS*4);
  f.read((uint8_t*)ow, OUTPUT_WEIGHTS*4);
  f.read((uint8_t*)ob, NUM_CLASSES*4);
  f.close();
  Serial.printf("%s weights loaded from SD\n", tag);
  return true;
}
bool myLoadAnnWeights() { return myLoadWeightsFile("/header/myWeights.bin", myConv1_w, myConv1_b, myConv2_w, myConv2_b, myOutput_w, myOutput_b, "ANN"); }
bool myLoadSnnWeights() { return myLoadWeightsFile("/header/mySnnWeights.bin", mySnnConv1_w, mySnnConv1_b, mySnnConv2_w, mySnnConv2_b, mySnnOutput_w, mySnnOutput_b, "SNN"); }

// ======================================================
// FORWARD DECLARATIONS
// ======================================================
void myActionCollect(int classIdx);
void myActionInferAnn();
void myActionInferSnn();
void myResetMenuState();
void myHandleMenuNavigation();
void myDrawMenu();
bool myCaptureAndPreprocess(float* dest);
void myAnnForward(float* input);
void mySnnForwardInfer();

// ======================================================
// PART 0: SETUP AND LOOP
// ======================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);
  Serial.println("\n=== XIAO ESP32-S3 Vision ML (ANN+SNN, real video, config-driven) Starting ===");
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  myRgbBuffer = (uint8_t*)ps_malloc(CAPTURE_SIZE*CAPTURE_SIZE*3);
  if (!myRgbBuffer) Serial.println("Failed to allocate RGB buffer!");

  pinMode(A0, INPUT);
  u8g2.begin();

  pinMode(21, OUTPUT); digitalWrite(21, HIGH); delay(100);
  Serial.println("Checking SD card...");
  SPI.begin(); SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card - continuing without it");
    u8g2.firstPage(); do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_240X240; config.jpeg_quality = 12;
  config.fb_count = 1;
  esp_camera_init(&config);
  Serial.println("Camera initialized");

  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) s->set_hmirror(s, 1);

  esp_log_level_set("*", ESP_LOG_WARN);
  esp_log_level_set("esp_camera", ESP_LOG_ERROR);

  if (!myLoadConfig()) { myApplyFallbackConfig(); myWriteDefaultConfigIfMissing(); }
  myComputeArchitecture();
  myBuildResizeLookup();
  myAllocateMemory();
  myAllocSnnWindow();

  #ifdef USE_BAKED_WEIGHTS
    #if defined(BAKED_NUM_CLASSES) && defined(BAKED_INPUT_SIZE) && defined(BAKED_NUM_CHANNELS)
      if (BAKED_NUM_CLASSES == NUM_CLASSES && BAKED_INPUT_SIZE == INPUT_SIZE && BAKED_NUM_CHANNELS == NUM_CHANNELS) {
        memcpy(myConv1_w, myModel_conv1_w, CONV1_WEIGHTS * sizeof(float));
        memcpy(myConv1_b, myModel_conv1_b, CONV1_FILTERS * sizeof(float));
        memcpy(myConv2_w, myModel_conv2_w, CONV2_WEIGHTS * sizeof(float));
        memcpy(myConv2_b, myModel_conv2_b, CONV2_FILTERS * sizeof(float));
        memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
        memcpy(myOutput_b, myModel_output_b, NUM_CLASSES * sizeof(float));
        myAnnTrained = true;
        memcpy(mySnnConv1_w, mySnnModel_conv1_w, CONV1_WEIGHTS * sizeof(float));
        memcpy(mySnnConv1_b, mySnnModel_conv1_b, CONV1_FILTERS * sizeof(float));
        memcpy(mySnnConv2_w, mySnnModel_conv2_w, CONV2_WEIGHTS * sizeof(float));
        memcpy(mySnnConv2_b, mySnnModel_conv2_b, CONV2_FILTERS * sizeof(float));
        memcpy(mySnnOutput_w, mySnnModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
        memcpy(mySnnOutput_b, mySnnModel_output_b, NUM_CLASSES * sizeof(float));
        mySnnTrained = true;
        Serial.println("Baked-in weights loaded (both models)");
      } else {
        Serial.println("Baked-in weights skipped: BAKED_* sizes don't match the current config.json.");
      }
    #else
      Serial.println("Baked-in weights skipped: header doesn't declare BAKED_NUM_CLASSES/BAKED_INPUT_SIZE/BAKED_NUM_CHANNELS to check against config.json.");
    #endif
  #endif

  if (myLoadAnnWeights()) { Serial.println("SD ANN weights in use"); myAnnTrained = true; }
  if (myLoadSnnWeights()) { Serial.println("SD SNN weights in use"); mySnnTrained = true; }

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);
  Serial.println("System ready - Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() { myHandleMenuNavigation(); }

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: VIDEO CLIP COLLECTION                                          ██
// ██  A tap now records a whole CLIP (SNN_TIMESTEPS real frames,             ██
// ██  CLIP_FRAME_INTERVAL_MS apart) into images/<folder>/clip_<id>/f00.jpg.. ██
// ██  matching index-v004.html's browser-side layout exactly.               ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████
void myRenderRgbToOLED(int overlayCount) {
  int myOledWidth = u8g2.getDisplayWidth();
  int myOledHeight = u8g2.getDisplayHeight();
  int myScaleX = CAPTURE_SIZE / myOledWidth;
  int myScaleY = CAPTURE_SIZE / myOledHeight;
  u8g2.firstPage();
  do {
    for (int myOledX = 0; myOledX < myOledWidth; myOledX++) {
      for (int myOledY = 0; myOledY < myOledHeight; myOledY++) {
        size_t myPixelIndex = ((myOledY * myScaleY) * CAPTURE_SIZE + (myOledX * myScaleX)) * 3;
        uint8_t myBrightness = (myRgbBuffer[myPixelIndex] + myRgbBuffer[myPixelIndex+1] + myRgbBuffer[myPixelIndex+2]) / 3;
        if (myBrightness > 100) u8g2.drawPixel(myOledX, myOledY);
      }
    }
    if (overlayCount >= 0) {
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0); u8g2.drawBox(0, 0, 20, 15);
      u8g2.setColorIndex(1); u8g2.setCursor(3, 10); u8g2.print(String(overlayCount));
    } else {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0); u8g2.drawBox(50, 0, 22, 8);
      u8g2.setColorIndex(1); u8g2.drawStr(52, 7, "LIVE");
    }
  } while (u8g2.nextPage());
}
void myDisplayImageOnOLED(camera_fb_t* fb, int overlayCount) {
  if (!myRgbBuffer) { Serial.println("RGB buffer not allocated - skipping OLED preview"); return; }
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) { Serial.println("Failed to convert JPEG to RGB888 for OLED"); return; }
  myRenderRgbToOLED(overlayCount);
}

// Counts existing samples in a class folder as CLIPS: a clip_* subfolder is
// one multi-frame clip; a loose .jpg directly in the folder (v002 layout,
// or something dropped in by hand) counts as one 1-frame clip - same
// convention as Store.listSamples() in index-v004.html.
int myCountClips(const String &classPath) {
  int n = 0;
  File root = SD.open(classPath);
  if (root) {
    File file = root.openNextFile();
    while (file) {
      String name = String(file.name());
      if (file.isDirectory()) n++;
      else if (name.endsWith(".jpg") || name.endsWith(".JPG")) n++;
      file.close();
      file = root.openNextFile();
    }
    root.close();
  }
  return n;
}

// Records one clip: SNN_TIMESTEPS real, sequential JPEG frames,
// CLIP_FRAME_INTERVAL_MS apart, into a fresh clip_<millis> subfolder.
void myCaptureClip(const String &classPath, int clipDisplayNumber) {
  String clipPath = classPath + "/clip_" + String(millis());
  SD.mkdir(clipPath);
  Serial.printf("  Recording clip (%d frames, %dms apart)...\n", SNN_TIMESTEPS, CLIP_FRAME_INTERVAL_MS);
  for (int i = 0; i < SNN_TIMESTEPS; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { Serial.println("  frame grab failed mid-clip"); continue; }
    char fname[16]; snprintf(fname, sizeof(fname), "/f%02d.jpg", i); // was char fname[8] - too small for "/fNN.jpg\0" (9 bytes), truncated to "/fNN.jp"
    File file = SD.open(clipPath + String(fname), FILE_WRITE);
    if (file) { file.write(fb->buf, fb->len); file.close(); }
    if (i == SNN_TIMESTEPS - 1) myDisplayImageOnOLED(fb, clipDisplayNumber);
    esp_camera_fb_return(fb);
    if (i < SNN_TIMESTEPS - 1) delay(CLIP_FRAME_INTERVAL_MS);
  }
}

void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect clips");
    u8g2.firstPage(); do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }
  const String &folder = myClassFolders[classIdx];
  Serial.printf("\n>>> Collection mode: %s (folder: %s)\n", myClassLabels[classIdx].c_str(), folder.c_str());
  Serial.printf("  TAP (1-2 taps) = Capture Clip (%d frames)\n", SNN_TIMESTEPS);
  Serial.println("  LONG PRESS (3+ taps) = Exit to menu");
  Serial.println("  Serial: 'T'=capture clip, 'L'=exit");
  myResetTouchState();

  String classPath = "/images/" + folder;
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(classPath)) SD.mkdir(classPath);

  int clipCount = myCountClips(classPath);

  unsigned long lastCameraDrain = 0, lastOLED = 0;
  bool oledNeedsUpdate = false, shouldCapture = false;
  while (true) {
    unsigned long now = millis();
    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;
      if (!shouldCapture) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (now - lastOLED > 250 && myRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) { oledNeedsUpdate = true; lastOLED = now; }
          }
          esp_camera_fb_return(fb);
        }
      }
    }
    if (oledNeedsUpdate) { oledNeedsUpdate = false; myRenderRgbToOLED(-1); }

    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'l' || c == 'L') { myResetMenuState(); return; }
      else if (c == 't' || c == 'T') shouldCapture = true;
    }
    int touchAction = myCheckTouchInput();
    if (touchAction == 2) { Serial.println("Exiting collection mode"); myResetMenuState(); return; }
    else if (touchAction == 1) shouldCapture = true;

    if (shouldCapture) {
      shouldCapture = false;
      myCaptureClip(classPath, clipCount + 1);
      clipCount++;
      Serial.printf("Clip saved for '%s' (total clips: %d)\n", myClassLabels[classIdx].c_str(), clipCount);
      lastOLED = millis();
    }
    delay(5);
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: FORWARD PASSES (ANN and SNN) - INFERENCE ONLY, no backward pass ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// Grabs one camera frame, decodes it, resizes to INPUT_SIZE, and fills
// `dest` with normalized 0-1 values (NUM_CHANNELS per pixel; if
// NUM_CHANNELS==1 the RGB frame is luminance-converted here). Also leaves
// myRgbBuffer holding the full 240x240 RGB frame for the OLED preview.
// `dest` lets the caller target either myInputBuffer (ANN / latest frame)
// or a slot in the SNN's rolling window directly.
bool myCaptureAndPreprocess(float* dest) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return false;
  bool ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer);
  if (ok) {
    for (int y=0;y<INPUT_SIZE;y++) {
      int sy = myResizeLookup[y];
      for (int x=0;x<INPUT_SIZE;x++) {
        int sx = myResizeLookup[x];
        int srcIdx = (sy*CAPTURE_SIZE+sx)*3;
        int dstIdx = (y*INPUT_SIZE+x)*NUM_CHANNELS;
        if (NUM_CHANNELS == 1) {
          float r=myRgbBuffer[srcIdx], g=myRgbBuffer[srcIdx+1], b=myRgbBuffer[srcIdx+2];
          dest[dstIdx] = (0.299f*r + 0.587f*g + 0.114f*b) * 0.003921569f;
        } else {
          dest[dstIdx]   = myRgbBuffer[srcIdx]   * 0.003921569f;
          dest[dstIdx+1] = myRgbBuffer[srcIdx+1] * 0.003921569f;
          dest[dstIdx+2] = myRgbBuffer[srcIdx+2] * 0.003921569f;
        }
      }
    }
  }
  esp_camera_fb_return(fb);
  return ok;
}

// ---- ANN forward pass: classifies whatever single frame `input` holds,
//      fills myDense_output[NUM_CLASSES] with softmax probabilities ----
void myAnnForward(float* input) {
  for (int f=0; f<CONV1_FILTERS; f++) {
    int ob = f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
    for (int y=0;y<CONV1_OUTPUT_SIZE;y++) for (int x=0;x<CONV1_OUTPUT_SIZE;x++) {
      float sum = myConv1_b[f];
      for (int ky=0;ky<CONV1_KERNEL_SIZE;ky++) for (int kx=0;kx<CONV1_KERNEL_SIZE;kx++) {
        int inPos = ((y+ky)*INPUT_SIZE+(x+kx))*NUM_CHANNELS;
        int wPos = f*CONV1_W_PER_FILTER + ky*CONV1_KERNEL_SIZE*NUM_CHANNELS + kx*NUM_CHANNELS;
        for (int c=0;c<NUM_CHANNELS;c++) sum += input[inPos+c]*myConv1_w[wPos+c];
      }
      myConv1_output[ob + y*CONV1_OUTPUT_SIZE + x] = leaky_relu(clip_value(sum));
    }
  }
  for (int f=0; f<CONV1_FILTERS; f++) {
    int ib=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, ob=f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
    for (int y=0;y<POOL1_OUTPUT_SIZE;y++) for (int x=0;x<POOL1_OUTPUT_SIZE;x++) {
      int iy=y*2, ix=x*2;
      float mv = myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix];
      mv = max(mv, myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix+1]);
      mv = max(mv, myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix]);
      mv = max(mv, myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1]);
      myPool1_output[ob+y*POOL1_OUTPUT_SIZE+x] = mv;
    }
  }
  for (int f=0; f<CONV2_FILTERS; f++) {
    int ob=f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
    for (int y=0;y<CONV2_OUTPUT_SIZE;y++) for (int x=0;x<CONV2_OUTPUT_SIZE;x++) {
      float sum = myConv2_b[f];
      for (int c=0;c<CONV1_FILTERS;c++) {
        int ib=c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
        for (int ky=0;ky<CONV2_KERNEL_SIZE;ky++) for (int kx=0;kx<CONV2_KERNEL_SIZE;kx++)
          sum += myPool1_output[ib+(y+ky)*POOL1_OUTPUT_SIZE+(x+kx)] * myConv2_w[f*CONV2_W_PER_FILTER+c*(CONV2_KERNEL_SIZE*CONV2_KERNEL_SIZE)+ky*CONV2_KERNEL_SIZE+kx];
      }
      myConv2_output[ob+y*CONV2_OUTPUT_SIZE+x] = leaky_relu(clip_value(sum));
    }
  }
  for (int c=0;c<NUM_CLASSES;c++) {
    double sum=0, comp=0;
    for (int i=0;i<FLATTENED_SIZE;i++) {
      double term = myConv2_output[i]*myOutput_w[c*FLATTENED_SIZE+i];
      double yv = term-comp, t = sum+yv;
      comp = (t-sum)-yv; sum = t;
    }
    myDense_output[c] = clip_value((float)sum + myOutput_b[c], -50, 50);
  }
  float mx = myDense_output[0]; for (int i=1;i<NUM_CLASSES;i++) mx = max(mx, myDense_output[i]);
  float expSum=0; for (int i=0;i<NUM_CLASSES;i++) expSum += exp(myDense_output[i]-mx);
  for (int i=0;i<NUM_CLASSES;i++) myDense_output[i] = exp(myDense_output[i]-mx)/expSum;
}

// Arranges the circular mySnnWindow buffer into oldest-first order, padding
// the front by repeating the oldest available real frame until the window
// has filled - same convention as the live-inference rolling window in
// index-v004.html.
void myBuildOrderedWindow(float** ordered) {
  int have = mySnnWindowFilled;
  int pad = SNN_TIMESTEPS - have;
  int oldestIdx = (mySnnWindowNext - have + SNN_TIMESTEPS) % SNN_TIMESTEPS;
  for (int i = 0; i < SNN_TIMESTEPS; i++) {
    int j = i - pad; if (j < 0) j = 0;
    ordered[i] = mySnnWindow[(oldestIdx + j) % SNN_TIMESTEPS];
  }
}

// ---- SNN forward pass (inference only): fills mySnnDense_output[NUM_CLASSES].
// Runs the FULL rolling window every call - NO EARLY EXIT (see the header
// comment's "WHAT CHANGED" #2: unlike a still image, every one of these
// SNN_TIMESTEPS real frames is genuinely new evidence, so there's nothing
// to skip). Each step rate-codes ONE REAL FRAME of the window into a
// Bernoulli spike frame, runs it through two LIF conv layers (membrane
// potential persists across steps), and accumulates a leaky per-class
// trace from the flattened Conv2 spikes; the final trace is softmaxed. ----
void mySnnForwardInfer() {
  static float** ordered = nullptr;
  if (!ordered) ordered = (float**)malloc(SNN_TIMESTEPS * sizeof(float*));
  myBuildOrderedWindow(ordered);

  memset(mySnnC1Mem, 0, CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*sizeof(float));
  memset(mySnnC2Mem, 0, CONV2_FILTERS*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*sizeof(float));
  memset(mySnnTrace, 0, NUM_CLASSES*sizeof(float));

  for (int t=0; t<SNN_TIMESTEPS; t++) {
    unsigned long stepStartUs = mySnnVerbose ? micros() : 0;
    float* frame = ordered[t];

    // 1) rate-code this REAL frame into a fresh Bernoulli spike frame
    for (int i=0;i<INPUT_N;i++)
      mySnnInputSpike[i] = (((float)random(0,10001))/10000.0f < frame[i]) ? 1 : 0;

    // 2) Conv1 LIF
    for (int f=0; f<CONV1_FILTERS; f++) {
      int ob = f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
      for (int y=0;y<CONV1_OUTPUT_SIZE;y++) for (int x=0;x<CONV1_OUTPUT_SIZE;x++) {
        float current = mySnnConv1_b[f];
        for (int ky=0;ky<CONV1_KERNEL_SIZE;ky++) for (int kx=0;kx<CONV1_KERNEL_SIZE;kx++) {
          int inPos = ((y+ky)*INPUT_SIZE+(x+kx))*NUM_CHANNELS;
          int wPos = f*CONV1_W_PER_FILTER + ky*CONV1_KERNEL_SIZE*NUM_CHANNELS + kx*NUM_CHANNELS;
          for (int c=0;c<NUM_CHANNELS;c++) current += mySnnInputSpike[inPos+c]*mySnnConv1_w[wPos+c];
        }
        int idx = ob + y*CONV1_OUTPUT_SIZE + x;
        float memPre = mySnnC1Mem[idx]*LIF_LEAK + current;
        uint8_t spike = memPre >= LIF_THRESHOLD ? 1 : 0;
        mySnnC1Mem[idx] = memPre - (spike ? LIF_THRESHOLD : 0);
        mySnnC1Spike[idx] = spike;
      }
    }

    // 3) Pool1: 2x2 OR of binary spikes
    for (int f=0; f<CONV1_FILTERS; f++) {
      int ib=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, ob=f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
      for (int y=0;y<POOL1_OUTPUT_SIZE;y++) for (int x=0;x<POOL1_OUTPUT_SIZE;x++) {
        int iy=y*2, ix=x*2;
        uint8_t sOr = mySnnC1Spike[ib+iy*CONV1_OUTPUT_SIZE+ix] |
                      mySnnC1Spike[ib+iy*CONV1_OUTPUT_SIZE+ix+1] |
                      mySnnC1Spike[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix] |
                      mySnnC1Spike[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1];
        mySnnPooled[ob+y*POOL1_OUTPUT_SIZE+x] = sOr;
      }
    }

    // 4) Conv2 LIF
    for (int f=0; f<CONV2_FILTERS; f++) {
      int ob = f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
      for (int y=0;y<CONV2_OUTPUT_SIZE;y++) for (int x=0;x<CONV2_OUTPUT_SIZE;x++) {
        float current = mySnnConv2_b[f];
        for (int c=0;c<CONV1_FILTERS;c++) {
          int ib=c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
          for (int ky=0;ky<CONV2_KERNEL_SIZE;ky++) for (int kx=0;kx<CONV2_KERNEL_SIZE;kx++)
            current += mySnnPooled[ib+(y+ky)*POOL1_OUTPUT_SIZE+(x+kx)] * mySnnConv2_w[f*CONV2_W_PER_FILTER+c*(CONV2_KERNEL_SIZE*CONV2_KERNEL_SIZE)+ky*CONV2_KERNEL_SIZE+kx];
        }
        int idx = ob + y*CONV2_OUTPUT_SIZE + x;
        float memPre = mySnnC2Mem[idx]*LIF_LEAK + current;
        uint8_t spike = memPre >= LIF_THRESHOLD ? 1 : 0;
        mySnnC2Mem[idx] = memPre - (spike ? LIF_THRESHOLD : 0);
        mySnnC2Spike[idx] = spike;
      }
    }

    // 5) Output: flatten Conv2 spikes -> per-class current -> leaky trace
    for (int c=0;c<NUM_CLASSES;c++) {
      float current = mySnnOutput_b[c];
      for (int i=0;i<FLATTENED_SIZE;i++) current += mySnnC2Spike[i]*mySnnOutput_w[c*FLATTENED_SIZE+i];
      mySnnTrace[c] = mySnnTrace[c]*SNN_TRACE_LEAK + current;
    }

    // Verbose: running softmax of the trace SO FAR plus this step's cost, so
    // you can watch the decision form. Display only - it never stops the run
    // early (see the header comment).
    if (mySnnVerbose) {
      memcpy(mySnnStepProbs, mySnnTrace, NUM_CLASSES*sizeof(float));
      float mxS = mySnnStepProbs[0]; for (int i=1;i<NUM_CLASSES;i++) mxS = max(mxS, mySnnStepProbs[i]);
      float esS = 0; for (int i=0;i<NUM_CLASSES;i++) esS += exp(mySnnStepProbs[i]-mxS);
      int best = 0;
      for (int i=0;i<NUM_CLASSES;i++) mySnnStepProbs[i] = exp(mySnnStepProbs[i]-mxS)/esS;
      for (int i=1;i<NUM_CLASSES;i++) if (mySnnStepProbs[i] > mySnnStepProbs[best]) best = i;
      int spikes1 = 0, spikes2 = 0;
      for (int i=0;i<CONV1_FILTERS*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;i++) spikes1 += mySnnC1Spike[i];
      for (int i=0;i<FLATTENED_SIZE;i++) spikes2 += mySnnC2Spike[i];
      Serial.printf("    [SNN t=%2d/%d] %5lums  conv1 spikes=%d conv2 spikes=%d  leading: %s p=%.2f\n",
                    t+1, SNN_TIMESTEPS, (micros()-stepStartUs)/1000UL, spikes1, spikes2,
                    myClassLabels[best].c_str(), mySnnStepProbs[best]);
    }
  }

  memcpy(mySnnDense_output, mySnnTrace, NUM_CLASSES*sizeof(float));
  float mx = mySnnDense_output[0]; for (int i=1;i<NUM_CLASSES;i++) mx = max(mx, mySnnDense_output[i]);
  float expSum=0; for (int i=0;i<NUM_CLASSES;i++) expSum += exp(mySnnDense_output[i]-mx);
  for (int i=0;i<NUM_CLASSES;i++) mySnnDense_output[i] = exp(mySnnDense_output[i]-mx)/expSum;
}

// Shared OLED+Serial display for a completed inference tick - used by both infer modes.
void myShowInferenceResult(const char* modelTag, float* probs, unsigned long frameMs) {
  int pred = 0;
  for (int i=1;i<NUM_CLASSES;i++) if (probs[i] > probs[pred]) pred = i;
  int oW = u8g2.getDisplayWidth(), oH = u8g2.getDisplayHeight();
  int scX = CAPTURE_SIZE/oW, scY = CAPTURE_SIZE/oH;
  u8g2.firstPage();
  do {
    for (int ox=0; ox<oW; ox++) for (int oy=0; oy<oH; oy++) {
      int pi = ((oy*scY)*CAPTURE_SIZE + (ox*scX))*3;
      uint8_t bright = (myRgbBuffer[pi]+myRgbBuffer[pi+1]+myRgbBuffer[pi+2])/3;
      if (bright > 100) u8g2.drawPixel(ox, oy);
    }
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.setColorIndex(0); u8g2.drawBox(0, oH-9, oW, 9);
    u8g2.setColorIndex(1);
    char buf[24];
    snprintf(buf, sizeof(buf), "%s %d%%", myClassLabels[pred].c_str(), (int)(probs[pred]*100));
    u8g2.drawStr(1, oH-1, buf);
  } while (u8g2.nextPage());
  Serial.printf("[%s] %.1fms Pred: %s (%.1f%%) | All:", modelTag, (float)frameMs, myClassLabels[pred].c_str(), probs[pred]*100);
  for (int i=0;i<NUM_CLASSES;i++) Serial.printf(" %.0f%%", probs[i]*100);
  Serial.println();
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE ACTIONS                                              ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████
void myActionInferAnn() {
  if (!myAnnTrained) {
    Serial.println("ERROR: No ANN weights loaded. Train in the browser and copy myWeights.bin to /header/ on the SD card.");
    u8g2.firstPage(); do { u8g2.drawStr(0,12,"No ANN weights"); u8g2.drawStr(0,24,"See browser"); } while (u8g2.nextPage());
    delay(2500); myResetMenuState(); return;
  }
  Serial.println("\n>>> Infer ANN (single latest frame, same as v002). Serial/touch: t or l = exit");
  myResetTouchState();
  int frameCount = 0;
  while (true) {
    if (Serial.available()) { char c = Serial.read(); if (c=='t'||c=='T'||c=='l'||c=='L') { myResetMenuState(); return; } }
    unsigned long t0 = millis();
    if (myCaptureAndPreprocess(myInputBuffer)) {
      myAnnForward(myInputBuffer);
      myShowInferenceResult("ANN", myDense_output, millis()-t0);
    }
    frameCount++;
    if (frameCount >= 5) {
      frameCount = 0;
      if (myReadTouch() > myThresholdPress) { delay(200); myResetMenuState(); return; }
    }
  }
}
void myActionInferSnn() {
  if (!mySnnTrained) {
    Serial.println("ERROR: No SNN weights loaded. Train in the browser and copy mySnnWeights.bin to /header/ on the SD card.");
    u8g2.firstPage(); do { u8g2.drawStr(0,12,"No SNN weights"); u8g2.drawStr(0,24,"See browser"); } while (u8g2.nextPage());
    delay(2500); myResetMenuState(); return;
  }
  Serial.println("\n>>> Infer SNN (real video - rolling window, full SNN_TIMESTEPS every tick, no early exit).");
  Serial.println("    Serial: 'v' = toggle verbose (per-timestep prints) / final result only,  't' or 'l' = exit");
  Serial.printf("    SNN_TIMESTEPS=%d LIF_LEAK=%.2f LIF_THRESHOLD=%.2f SNN_TRACE_LEAK=%.2f CLIP_FRAME_INTERVAL_MS=%d (from config.json)\n",
                SNN_TIMESTEPS, LIF_LEAK, LIF_THRESHOLD, SNN_TRACE_LEAK, CLIP_FRAME_INTERVAL_MS);
  Serial.printf("    Verbose is %s. The first %d ticks fill the window (padded with the oldest frame), so a fully\n"
                "    real-video decision needs ~%d ticks - that fill time, not one slow pass, is most of any long delay.\n",
                mySnnVerbose ? "ON" : "OFF", SNN_TIMESTEPS, SNN_TIMESTEPS);
  myResetTouchState();
  mySnnWindowFilled = 0; mySnnWindowNext = 0; // start the window fresh each time inference mode is entered
  int frameCount = 0, tick = 0;
  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c=='t'||c=='T'||c=='l'||c=='L') { myResetMenuState(); return; }
      if (c=='v'||c=='V') { mySnnVerbose = !mySnnVerbose; Serial.printf("    Verbose %s\n", mySnnVerbose ? "ON" : "OFF"); }
    }
    unsigned long t0 = millis();
    bool ok = myCaptureAndPreprocess(mySnnWindow[mySnnWindowNext]);
    unsigned long capMs = millis() - t0;
    if (ok) {
      mySnnWindowNext = (mySnnWindowNext + 1) % SNN_TIMESTEPS;
      if (mySnnWindowFilled < SNN_TIMESTEPS) mySnnWindowFilled++;
      tick++;
      if (mySnnVerbose) Serial.printf("  [SNN tick %d] window %d/%d real frames, capture %lums, running %d timesteps...\n",
                                      tick, mySnnWindowFilled, SNN_TIMESTEPS, capMs, SNN_TIMESTEPS);
      unsigned long t1 = millis();
      mySnnForwardInfer();
      unsigned long infMs = millis() - t1;
      myShowInferenceResult("SNN", mySnnDense_output, millis()-t0);
      if (mySnnVerbose) Serial.printf("    capture=%lums inference=%lums total=%lums%s\n", capMs, infMs, millis()-t0,
                                      mySnnWindowFilled < SNN_TIMESTEPS ? "  (window still filling)" : "");
    }
    frameCount++;
    if (frameCount >= 5) {
      frameCount = 0;
      if (myReadTouch() > myThresholdPress) { delay(200); myResetMenuState(); return; }
    }
    // Pace captures to the clip interval the model was trained on, minus
    // however long this tick's capture+inference already took.
    long remaining = (long)CLIP_FRAME_INTERVAL_MS - (long)(millis() - t0);
    if (remaining > 0) delay(remaining);
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: MENU SYSTEM FUNCTIONS - class list/count now dynamic           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████
String myMenuLabel(int i) {
  if (i <= NUM_CLASSES) return myClassLabels[i-1];
  return (i == NUM_CLASSES+1) ? "Infer ANN" : "Infer SNN";
}
void myResetMenuState() { myIsSelected = false; myResetTouchState(); myLastActivityTime = millis(); myDrawMenu(); }

void myDrawMenu() {
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    Serial.print(i == myMenuIndex ? " > " : "   ");
    Serial.printf("%d. %s\n", i, myMenuLabel(i).c_str());
  }
  Serial.println("Commands: t=next (tap)  l=select (longpress)");
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
    int myStartItem = (myMenuIndex <= NUM_CLASSES) ? 1 : myMenuIndex - 2;
    for (int i=0;i<3;i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;
      int y = 18 + i*9;
      String label = myMenuLabel(cur);
      u8g2.drawStr(0, y, (cur == myMenuIndex ? ("> "+label) : ("  "+label)).c_str());
    }
  } while (u8g2.nextPage());
}
void myExecuteMenuItem(int idx) {
  if (idx <= NUM_CLASSES) myActionCollect(idx-1);
  else if (idx == NUM_CLASSES+1) myActionInferAnn();
  else myActionInferSnn();
}
void myHandleMenuNavigation() {
  unsigned long now = millis();
  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();
    if (c >= '1' && c <= '9') {
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) { myMenuIndex = newIndex; myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex); }
    } else if (c == 't' || c == 'T') {
      if (now - myLastTapTime > myTapCooldown) { myMenuIndex++; if (myMenuIndex > myTotalItems) myMenuIndex = 1; myDrawMenu(); myLastTapTime = now; myLastActivityTime = now; }
    } else if (c == 'l' || c == 'L') {
      myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
    }
  }
  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();
    if (touchAction == 1) {
      if (now - myLastTapTime > myTapCooldown) { myMenuIndex++; if (myMenuIndex > myTotalItems) myMenuIndex = 1; myDrawMenu(); myLastTapTime = now; myLastActivityTime = now; }
    } else if (touchAction == 2) {
      myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
    }
  }
}
