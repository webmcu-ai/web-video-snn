/*
======================================================================
VISION ANN/SNN - ON-DEVICE VIDEO ANALYSIS (XIAO ESP32S3)
firmware-v003.ino

Companion to index-v003.html. Loads myWeights.bin (ANN) and
mySnnWeights.bin (SNN) from the SD card's /header/ folder - files are
byte-identical in layout to what the browser trainer writes (float32
little-endian: conv1_w, conv1_b, conv2_w, conv2_b, output_w, output_b,
in that order), so weights trained in the browser just work here.

WHAT THIS IS: a rewrite from the reference math in index-v003.html, not
an edit of the firmware-v002.ino you attached - that file is the raw
"record video to an .avi on the SD card" sketch and does not contain any
of the myAnnForward / mySnnForwardInfer / weight-loading code the
webpage's "Device settings" panel and byte-compatible .bin files assume
exists on the device. If you have that actual on-device ML sketch
(with your OLED/touch-menu/baked-header conventions) as a separate file,
send it over and this can be merged into it properly instead of living
as a new, simpler sketch. As written, this sketch only does capture +
classify + Serial output - no OLED, no baked-header fallback tier, no
SD video recording.

ARCHITECTURE (must match index-v003.html exactly - see computeArch() there)
  Conv1(3x3,4 filters) -> Pool(2x2 max) -> Conv2(3x3,8 filters) -> flatten -> Dense -> softmax
  ANN: ordinary leaky-ReLU neurons - classifies the single latest frame.
  SNN: every neuron above is a spiking LIF neuron, run for SNN_TIMESTEPS
       steps. Each step now consumes one REAL frame from a rolling window
       of the last SNN_TIMESTEPS camera frames (captured
       CLIP_FRAME_INTERVAL_MS apart) - exactly mirroring the browser's
       live SNN inference in index-v003.html. This is the actual "video
       analysis": the LIF membrane potentials integrate genuine motion
       across real time, not a repeated still image.

CAMERA: FRAMESIZE_240X240 gives a square frame directly (no cropping
needed), decoded from JPEG to RGB888 via the esp32-camera library's
fmt2rgb888(), then box-filter downsampled to INPUT_SIZE x INPUT_SIZE and
normalized to [0,1] - matching fillInputBufferFromImageData() in the
webpage (grayscale: buf[p] = 0.299R+0.587G+0.114B; RGB: buf[p*3+c]).
Downsampling here is a box average rather than the browser canvas's
bilinear resize, so expect a small train/inference domain shift - if
accuracy on-device looks worse than in the browser, that mismatch is
the first thing to suspect.

PASTE FROM THE WEBPAGE'S "Device settings" PANEL: copy INPUT_SIZE,
NUM_CHANNELS, NUM_CLASSES, myClassLabels[], SNN_TIMESTEPS, LIF_LEAK,
LIF_THRESHOLD, SNN_TRACE_LEAK, and CLIP_FRAME_INTERVAL_MS from there into
the block below - they must match exactly or the .bin files will fail
their size check and refuse to load (same safety check the webpage uses).

By Jeremy Ellis, with Claude's help porting the browser's CNN/SNN math
to C++ for on-device video inference. Use at your own risk! MIT license.
======================================================================
*/

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"

// ---------------------------------------------------------------------
// CAMERA PINS - XIAO ESP32S3 Sense (unchanged from firmware-v002.ino)
// ---------------------------------------------------------------------
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13
const int SD_PIN_CS = 21;
#define CAPTURE_SIZE      240   // FRAMESIZE_240X240 - square, no cropping needed

// ---------------------------------------------------------------------
// PASTE THESE FROM index-v003.html's "Device settings" PANEL
// (values below are just the webpage's defaults - replace with yours)
// ---------------------------------------------------------------------
#define INPUT_SIZE 64
#define NUM_CHANNELS 3
#define NUM_CLASSES 2
String myClassLabels[NUM_CLASSES] = {"class_a", "class_b"};
#define SNN_TIMESTEPS 16              // also: real frames in the rolling window
float LIF_LEAK = 0.90f;
float LIF_THRESHOLD = 0.50f;
float SNN_TRACE_LEAK = 0.95f;
#define CLIP_FRAME_INTERVAL_MS 80     // target gap between real frames

// ---------------------------------------------------------------------
// FIXED ARCHITECTURE CONSTANTS - match computeArch() in index-v003.html.
// The device does not make these configurable.
// ---------------------------------------------------------------------
#define CONV1_KERNEL 3
#define CONV1_FILTERS 4
#define CONV2_KERNEL 3
#define CONV2_FILTERS 8
const int CONV1_OUT = INPUT_SIZE - (CONV1_KERNEL - 1);
const int POOL1_OUT = CONV1_OUT / 2;
const int CONV2_OUT = POOL1_OUT - (CONV2_KERNEL - 1);
const int FLAT = CONV2_OUT * CONV2_OUT * CONV2_FILTERS;
const int CONV1_W_PER_FILTER = CONV1_KERNEL * CONV1_KERNEL * NUM_CHANNELS;
const int CONV1_WEIGHTS = CONV1_W_PER_FILTER * CONV1_FILTERS;
const int CONV2_W_PER_FILTER = CONV2_KERNEL * CONV2_KERNEL * CONV1_FILTERS;
const int CONV2_WEIGHTS = CONV2_W_PER_FILTER * CONV2_FILTERS;
const int OUTPUT_WEIGHTS = FLAT * NUM_CLASSES;
const int INPUT_N = INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS;

// ---------------------------------------------------------------------
// WEIGHT STORAGE - one flat struct per model, loaded straight from the
// .bin file in the exact float32 order modelSerialize() writes it in.
// ---------------------------------------------------------------------
struct ModelWeights {
  float *conv1_w, *conv1_b, *conv2_w, *conv2_b, *output_w, *output_b;
  bool loaded = false;
};
ModelWeights Ann, Snn;

void allocModel(ModelWeights &m) {
  m.conv1_w   = (float*)malloc(CONV1_WEIGHTS * sizeof(float));
  m.conv1_b   = (float*)malloc(CONV1_FILTERS * sizeof(float));
  m.conv2_w   = (float*)malloc(CONV2_WEIGHTS * sizeof(float));
  m.conv2_b   = (float*)malloc(CONV2_FILTERS * sizeof(float));
  m.output_w  = (float*)malloc(OUTPUT_WEIGHTS * sizeof(float));
  m.output_b  = (float*)malloc(NUM_CLASSES * sizeof(float));
}
// Fresh (unloaded) fallback so the device doesn't crash without a .bin,
// even though its predictions will be meaningless - mirrors the
// He-init fallback in the webpage's rebuildArchitecture(). Add your usual
// baked-header tier here too if you keep one (SD > baked-header > this).
void heInitModel(ModelWeights &m) {
  randomSeed(esp_random());
  float c1 = sqrtf(2.0f / CONV1_W_PER_FILTER);
  for (int i = 0; i < CONV1_WEIGHTS; i++) m.conv1_w[i] = ((float)random(-1000,1000)/1000.0f) * c1;
  for (int i = 0; i < CONV1_FILTERS; i++) m.conv1_b[i] = 0;
  float c2 = sqrtf(2.0f / CONV2_W_PER_FILTER);
  for (int i = 0; i < CONV2_WEIGHTS; i++) m.conv2_w[i] = ((float)random(-1000,1000)/1000.0f) * c2;
  for (int i = 0; i < CONV2_FILTERS; i++) m.conv2_b[i] = 0;
  float d = sqrtf(2.0f / max(FLAT,1));
  for (int i = 0; i < OUTPUT_WEIGHTS; i++) m.output_w[i] = ((float)random(-1000,1000)/1000.0f) * d;
  for (int i = 0; i < NUM_CLASSES; i++) m.output_b[i] = 0;
}
// Loads a .bin exactly like annDeserialize/snnDeserialize in the webpage:
// same field order, same size check (wrong size = refuse, keep old weights).
bool loadWeightsFromSd(const char *path, ModelWeights &m) {
  size_t totalFloats = CONV1_WEIGHTS + CONV1_FILTERS + CONV2_WEIGHTS + CONV2_FILTERS + OUTPUT_WEIGHTS + NUM_CLASSES;
  File f = SD.open(path, FILE_READ);
  if (!f) { Serial.printf("[weights] %s not found.\n", path); return false; }
  if (f.size() != totalFloats * 4) {
    Serial.printf("[weights] %s size %u != expected %u - architecture mismatch, not loaded.\n", path, (unsigned)f.size(), (unsigned)(totalFloats*4));
    f.close(); return false;
  }
  auto readArr = [&](float *arr, int n) { f.read((uint8_t*)arr, n * sizeof(float)); };
  readArr(m.conv1_w, CONV1_WEIGHTS);  readArr(m.conv1_b, CONV1_FILTERS);
  readArr(m.conv2_w, CONV2_WEIGHTS);  readArr(m.conv2_b, CONV2_FILTERS);
  readArr(m.output_w, OUTPUT_WEIGHTS); readArr(m.output_b, NUM_CLASSES);
  f.close();
  m.loaded = true;
  Serial.printf("[weights] %s loaded.\n", path);
  return true;
}

// ---------------------------------------------------------------------
// MATH HELPERS - must match the webpage exactly (leakyRelu, clipValue,
// softmax, surrogate not needed on-device since the device never trains).
// ---------------------------------------------------------------------
inline float leakyRelu(float x) { return x > 0 ? x : 0.1f * x; }
inline float clipValue(float v, float mn = -100, float mx = 100) {
  if (isnan(v) || isinf(v)) return 0;
  return v < mn ? mn : (v > mx ? mx : v);
}
void softmaxInPlace(float *arr, int n) {
  float mx = arr[0]; for (int i = 1; i < n; i++) if (arr[i] > mx) mx = arr[i];
  float sum = 0; for (int i = 0; i < n; i++) { arr[i] = expf(arr[i] - mx); sum += arr[i]; }
  for (int i = 0; i < n; i++) arr[i] /= sum;
}
inline float randUnit() { return (float)esp_random() / (float)UINT32_MAX; } // [0,1)

// ---------------------------------------------------------------------
// ANN FORWARD - ported 1:1 from annForward() in index-v003.html.
// ---------------------------------------------------------------------
float *conv1Out, *pool1Out, *conv2Out, *annProbs;
void allocAnnBuffers() {
  conv1Out = (float*)malloc(CONV1_FILTERS * CONV1_OUT * CONV1_OUT * sizeof(float));
  pool1Out = (float*)malloc(CONV1_FILTERS * POOL1_OUT * POOL1_OUT * sizeof(float));
  conv2Out = (float*)malloc(FLAT * sizeof(float));
  annProbs = (float*)malloc(NUM_CLASSES * sizeof(float));
}
void annForward(float *input) {
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUT * CONV1_OUT;
    for (int y = 0; y < CONV1_OUT; y++) for (int x = 0; x < CONV1_OUT; x++) {
      float sum = Ann.conv1_b[f];
      for (int ky = 0; ky < CONV1_KERNEL; ky++) for (int kx = 0; kx < CONV1_KERNEL; kx++) {
        int inPos = ((y+ky)*INPUT_SIZE + (x+kx)) * NUM_CHANNELS;
        int wPos = f*CONV1_W_PER_FILTER + ky*CONV1_KERNEL*NUM_CHANNELS + kx*NUM_CHANNELS;
        for (int c = 0; c < NUM_CHANNELS; c++) sum += input[inPos+c] * Ann.conv1_w[wPos+c];
      }
      conv1Out[ob + y*CONV1_OUT + x] = leakyRelu(clipValue(sum));
    }
  }
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ib = f*CONV1_OUT*CONV1_OUT, ob = f*POOL1_OUT*POOL1_OUT;
    for (int y = 0; y < POOL1_OUT; y++) for (int x = 0; x < POOL1_OUT; x++) {
      int iy=y*2, ix=x*2;
      float mv = conv1Out[ib+iy*CONV1_OUT+ix];
      mv = max(mv, conv1Out[ib+iy*CONV1_OUT+ix+1]);
      mv = max(mv, conv1Out[ib+(iy+1)*CONV1_OUT+ix]);
      mv = max(mv, conv1Out[ib+(iy+1)*CONV1_OUT+ix+1]);
      pool1Out[ob + y*POOL1_OUT + x] = mv;
    }
  }
  for (int f = 0; f < CONV2_FILTERS; f++) {
    int ob = f*CONV2_OUT*CONV2_OUT;
    for (int y = 0; y < CONV2_OUT; y++) for (int x = 0; x < CONV2_OUT; x++) {
      float sum = Ann.conv2_b[f];
      for (int c = 0; c < CONV1_FILTERS; c++) {
        int ib = c*POOL1_OUT*POOL1_OUT;
        for (int ky = 0; ky < CONV2_KERNEL; ky++) for (int kx = 0; kx < CONV2_KERNEL; kx++)
          sum += pool1Out[ib+(y+ky)*POOL1_OUT+(x+kx)] * Ann.conv2_w[f*CONV2_W_PER_FILTER + c*(CONV2_KERNEL*CONV2_KERNEL) + ky*CONV2_KERNEL + kx];
      }
      conv2Out[ob + y*CONV2_OUT + x] = leakyRelu(clipValue(sum));
    }
  }
  for (int c = 0; c < NUM_CLASSES; c++) {
    float sum = Ann.output_b[c];
    for (int i = 0; i < FLAT; i++) sum += conv2Out[i] * Ann.output_w[c*FLAT+i];
    annProbs[c] = clipValue(sum, -50, 50);
  }
  softmaxInPlace(annProbs, NUM_CLASSES);
}

// ---------------------------------------------------------------------
// SNN FORWARD - ported 1:1 from the REAL-VIDEO snnForward() in
// index-v003.html: `frames` is SNN_TIMESTEPS real camera frames (the
// rolling window filled in loop()), one real frame consumed per timestep.
// ---------------------------------------------------------------------
float *snnC1Mem, *snnC2Mem, *snnTrace, *snnProbs;
uint8_t *snnInputSpike, *snnC1Spike, *snnPooled, *snnC2Spike;
void allocSnnBuffers() {
  snnC1Mem = (float*)malloc(CONV1_FILTERS*CONV1_OUT*CONV1_OUT*sizeof(float));
  snnC2Mem = (float*)malloc(FLAT*sizeof(float));
  snnTrace = (float*)malloc(NUM_CLASSES*sizeof(float));
  snnProbs = (float*)malloc(NUM_CLASSES*sizeof(float));
  snnInputSpike = (uint8_t*)malloc(INPUT_N);
  snnC1Spike = (uint8_t*)malloc(CONV1_FILTERS*CONV1_OUT*CONV1_OUT);
  snnPooled = (uint8_t*)malloc(CONV1_FILTERS*POOL1_OUT*POOL1_OUT);
  snnC2Spike = (uint8_t*)malloc(FLAT);
}
// frames: array of SNN_TIMESTEPS pointers, each to an INPUT_N float buffer.
void snnForwardInfer(float **frames) {
  memset(snnC1Mem, 0, CONV1_FILTERS*CONV1_OUT*CONV1_OUT*sizeof(float));
  memset(snnC2Mem, 0, FLAT*sizeof(float));
  memset(snnTrace, 0, NUM_CLASSES*sizeof(float));

  for (int t = 0; t < SNN_TIMESTEPS; t++) {
    float *frame = frames[t];
    for (int i = 0; i < INPUT_N; i++) snnInputSpike[i] = (randUnit() < frame[i]) ? 1 : 0;

    for (int f = 0; f < CONV1_FILTERS; f++) {
      int ob = f*CONV1_OUT*CONV1_OUT;
      for (int y = 0; y < CONV1_OUT; y++) for (int x = 0; x < CONV1_OUT; x++) {
        float current = Snn.conv1_b[f];
        for (int ky = 0; ky < CONV1_KERNEL; ky++) for (int kx = 0; kx < CONV1_KERNEL; kx++) {
          int inPos = ((y+ky)*INPUT_SIZE + (x+kx)) * NUM_CHANNELS;
          int wPos = f*CONV1_W_PER_FILTER + ky*CONV1_KERNEL*NUM_CHANNELS + kx*NUM_CHANNELS;
          for (int c = 0; c < NUM_CHANNELS; c++) current += snnInputSpike[inPos+c] * Snn.conv1_w[wPos+c];
        }
        int idx = ob + y*CONV1_OUT + x;
        float memPre = snnC1Mem[idx]*LIF_LEAK + current;
        uint8_t spike = memPre >= LIF_THRESHOLD ? 1 : 0;
        snnC1Mem[idx] = memPre - (spike ? LIF_THRESHOLD : 0);
        snnC1Spike[idx] = spike;
      }
    }
    for (int f = 0; f < CONV1_FILTERS; f++) {
      int ib = f*CONV1_OUT*CONV1_OUT, ob = f*POOL1_OUT*POOL1_OUT;
      for (int y = 0; y < POOL1_OUT; y++) for (int x = 0; x < POOL1_OUT; x++) {
        int iy=y*2, ix=x*2;
        uint8_t sOr = snnC1Spike[ib+iy*CONV1_OUT+ix] | snnC1Spike[ib+iy*CONV1_OUT+ix+1] |
                      snnC1Spike[ib+(iy+1)*CONV1_OUT+ix] | snnC1Spike[ib+(iy+1)*CONV1_OUT+ix+1];
        snnPooled[ob + y*POOL1_OUT + x] = sOr;
      }
    }
    for (int f = 0; f < CONV2_FILTERS; f++) {
      int ob = f*CONV2_OUT*CONV2_OUT;
      for (int y = 0; y < CONV2_OUT; y++) for (int x = 0; x < CONV2_OUT; x++) {
        float current = Snn.conv2_b[f];
        for (int c = 0; c < CONV1_FILTERS; c++) {
          int ib = c*POOL1_OUT*POOL1_OUT;
          for (int ky = 0; ky < CONV2_KERNEL; ky++) for (int kx = 0; kx < CONV2_KERNEL; kx++)
            current += snnPooled[ib+(y+ky)*POOL1_OUT+(x+kx)] * Snn.conv2_w[f*CONV2_W_PER_FILTER + c*(CONV2_KERNEL*CONV2_KERNEL) + ky*CONV2_KERNEL + kx];
        }
        int idx = ob + y*CONV2_OUT + x;
        float memPre = snnC2Mem[idx]*LIF_LEAK + current;
        uint8_t spike = memPre >= LIF_THRESHOLD ? 1 : 0;
        snnC2Mem[idx] = memPre - (spike ? LIF_THRESHOLD : 0);
        snnC2Spike[idx] = spike;
      }
    }
    for (int c = 0; c < NUM_CLASSES; c++) {
      float current = Snn.output_b[c];
      for (int i = 0; i < FLAT; i++) current += snnC2Spike[i] * Snn.output_w[c*FLAT+i];
      snnTrace[c] = snnTrace[c]*SNN_TRACE_LEAK + current;
    }
  }
  memcpy(snnProbs, snnTrace, NUM_CLASSES*sizeof(float));
  softmaxInPlace(snnProbs, NUM_CLASSES);
}

// ---------------------------------------------------------------------
// CAMERA - grab a JPEG, decode to RGB888, box-filter downsample to
// INPUT_SIZE, normalize to [0,1] - matches fillInputBufferFromImageData().
// ---------------------------------------------------------------------
bool captureInputFrame(float *out) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { Serial.println("[camera] frame grab failed"); return false; }
  static uint8_t *rgb = nullptr;
  if (!rgb) rgb = (uint8_t*)malloc(CAPTURE_SIZE * CAPTURE_SIZE * 3);
  bool ok = fmt2rgb888(fb->buf, fb->len, fb->format, rgb);
  esp_camera_fb_return(fb);
  if (!ok) { Serial.println("[camera] JPEG decode failed"); return false; }

  const float block = (float)CAPTURE_SIZE / INPUT_SIZE;
  for (int y = 0; y < INPUT_SIZE; y++) {
    for (int x = 0; x < INPUT_SIZE; x++) {
      int sy0 = (int)(y*block), sy1 = max(sy0+1, (int)((y+1)*block));
      int sx0 = (int)(x*block), sx1 = max(sx0+1, (int)((x+1)*block));
      long rs=0, gs=0, bs=0; int n=0;
      for (int sy = sy0; sy < sy1 && sy < CAPTURE_SIZE; sy++)
        for (int sx = sx0; sx < sx1 && sx < CAPTURE_SIZE; sx++) {
          int p = (sy*CAPTURE_SIZE + sx) * 3;
          rs += rgb[p]; gs += rgb[p+1]; bs += rgb[p+2]; n++;
        }
      float r = (rs/(float)n)/255.0f, g = (gs/(float)n)/255.0f, b = (bs/(float)n)/255.0f;
      int p = y*INPUT_SIZE + x;
      if (NUM_CHANNELS == 1) out[p] = 0.299f*r + 0.587f*g + 0.114f*b;
      else { out[p*3]=r; out[p*3+1]=g; out[p*3+2]=b; }
    }
  }
  return true;
}

// ---------------------------------------------------------------------
// SETUP / LOOP
// ---------------------------------------------------------------------
float *rollingWindow[SNN_TIMESTEPS]; // circular buffer of real frames
int windowFilled = 0, windowNext = 0;

void setup() {
  Serial.begin(115200);
  delay(200);

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0=Y2_GPIO_NUM; config.pin_d1=Y3_GPIO_NUM; config.pin_d2=Y4_GPIO_NUM; config.pin_d3=Y5_GPIO_NUM;
  config.pin_d4=Y6_GPIO_NUM; config.pin_d5=Y7_GPIO_NUM; config.pin_d6=Y8_GPIO_NUM; config.pin_d7=Y9_GPIO_NUM;
  config.pin_xclk=XCLK_GPIO_NUM; config.pin_pclk=PCLK_GPIO_NUM; config.pin_vsync=VSYNC_GPIO_NUM; config.pin_href=HREF_GPIO_NUM;
  config.pin_sscb_sda=SIOD_GPIO_NUM; config.pin_sscb_scl=SIOC_GPIO_NUM;
  config.pin_pwdn=PWDN_GPIO_NUM; config.pin_reset=RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_240X240;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2; // double-buffer so grabbing doesn't stall the sensor

  if (esp_camera_init(&config) != ESP_OK) { Serial.println("Camera init failed"); return; }
  if (!SD.begin(SD_PIN_CS)) { Serial.println("SD init failed - weights won't load, using random init."); }

  allocModel(Ann); allocModel(Snn);
  allocAnnBuffers(); allocSnnBuffers();
  if (!loadWeightsFromSd("/header/myWeights.bin", Ann)) heInitModel(Ann);
  if (!loadWeightsFromSd("/header/mySnnWeights.bin", Snn)) heInitModel(Snn);

  for (int t = 0; t < SNN_TIMESTEPS; t++) rollingWindow[t] = (float*)malloc(INPUT_N*sizeof(float));

  Serial.printf("Ready. %dx%dx%d, %d classes, %d SNN timesteps @ %dms.\n",
                INPUT_SIZE, INPUT_SIZE, NUM_CHANNELS, NUM_CLASSES, SNN_TIMESTEPS, CLIP_FRAME_INTERVAL_MS);
}

void loop() {
  static float *latest = (float*)malloc(INPUT_N*sizeof(float));
  if (!captureInputFrame(latest)) { delay(100); return; }

  // Push into the rolling window (real frames, real time axis).
  memcpy(rollingWindow[windowNext], latest, INPUT_N*sizeof(float));
  windowNext = (windowNext + 1) % SNN_TIMESTEPS;
  if (windowFilled < SNN_TIMESTEPS) windowFilled++;

  // ANN: classify the single latest frame.
  annForward(latest);
  int annPred = 0; for (int i=1;i<NUM_CLASSES;i++) if (annProbs[i]>annProbs[annPred]) annPred=i;

  // SNN: classify the real video in the rolling window, oldest-first.
  // Before the window fills, pad the front by repeating the oldest frame
  // we have (same convention as the webpage's live inference).
  float *ordered[SNN_TIMESTEPS];
  int have = windowFilled;
  int pad = SNN_TIMESTEPS - have;
  int oldestIdx = (windowNext - have + SNN_TIMESTEPS) % SNN_TIMESTEPS;
  for (int i = 0; i < SNN_TIMESTEPS; i++) {
    int j = i - pad; if (j < 0) j = 0;
    ordered[i] = rollingWindow[(oldestIdx + j) % SNN_TIMESTEPS];
  }
  snnForwardInfer(ordered);
  int snnPred = 0; for (int i=1;i<NUM_CLASSES;i++) if (snnProbs[i]>snnProbs[snnPred]) snnPred=i;

  Serial.printf("[ANN] %s (%.0f%%)   [SNN] %s (%.0f%%)\n",
                myClassLabels[annPred].c_str(), annProbs[annPred]*100.0f,
                myClassLabels[snnPred].c_str(), snnProbs[snnPred]*100.0f);

  delay(CLIP_FRAME_INTERVAL_MS);
}
