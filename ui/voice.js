/*
  Browser audio for Maslul: microphone capture and counterpart playback.

  WHY AudioWorklet AND NOT MediaRecorder: MediaRecorder produces encoded
  chunks (webm/opus) on its own schedule, typically 100ms+ apart, and the
  server would have to decode them. An AudioWorklet gives raw PCM at a
  fixed small interval, which is what both the STT API and the turn
  detector want -- and the small interval is what makes barge-in detection
  prompt rather than arriving a tenth of a second late.

  WHY A SCHEDULED PLAYBACK QUEUE AND NOT <audio>: the counterpart's audio
  arrives as a stream of PCM chunks that must play gap-free, and must be
  DISCARDABLE the instant he is interrupted. An <audio> element buffers
  internally, so cutting him off would leave queued audio still playing --
  the exact talk-over the whole design forbids. Scheduling buffers against
  the AudioContext clock gives gap-free playback and an instant flush.
*/

const SAMPLE_RATE = 16000;

// 20ms frames. Small enough that barge-in is detected promptly, large
// enough that per-message overhead stays irrelevant.
const FRAME_SAMPLES = 320;

// The worklet runs on the audio thread and cannot allocate freely, so it
// fills a fixed buffer and posts it when full.
const WORKLET_SOURCE = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Int16Array(${FRAME_SAMPLES});
    this.offset = 0;
  }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      // Float [-1,1] to signed 16-bit, clamped: values can exceed the
      // nominal range and would wrap to loud noise otherwise.
      let s = Math.max(-1, Math.min(1, channel[i]));
      this.buffer[this.offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.offset === this.buffer.length) {
        this.port.postMessage(this.buffer.slice());
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture', CaptureProcessor);
`;

export class VoiceClient {
  constructor({ sessionId, apiBase = "", onEvent = () => {}, endpoint = "voice" }) {
    this.sessionId = sessionId;
    this.apiBase = apiBase;
    this.onEvent = onEvent;
    // "voice" = cascade (STT -> agent -> TTS), "live" = Gemini native
    // speech-to-speech. They differ in output sample rate, which the
    // server announces on connect.
    this.endpoint = endpoint;
    this.outputRate = SAMPLE_RATE;

    this.socket = null;
    this.micContext = null;
    this.playContext = null;
    this.stream = null;

    // Next moment, on the playback clock, at which audio should start.
    // Tracked explicitly so consecutive chunks butt up against each other
    // instead of leaving audible gaps.
    this.playCursor = 0;
    this.scheduled = new Set();
    this.muted = false;
  }

  async connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const host = this.apiBase ? this.apiBase.replace(/^https?:\/\//, "") : location.host;
    this.socket = new WebSocket(
      `${scheme}://${host}/sessions/${this.sessionId}/${this.endpoint}`);
    this.socket.binaryType = "arraybuffer";

    this.socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        const message = JSON.parse(event.data);
        if (message.type === "flush_audio") {
          // He was cut off: drop everything queued, immediately.
          this.flushPlayback();
        }
        if (message.type === "voice_ready" && message.output_sample_rate) {
          // Gemini Live outputs 24kHz while the cascade outputs 16kHz.
          // Playing at the wrong rate makes the voice sound chipmunked or
          // slurred, so the rate is taken from the server rather than
          // assumed.
          this.outputRate = message.output_sample_rate;
          this.rebuildPlayback();
        }
        this.onEvent(message);
      } else {
        this.safeEnqueue(event.data);
      }
    };

    this.socket.onclose = () => this.onEvent({ type: "voice_closed" });
    this.socket.onerror = () => this.onEvent({ type: "voice_error" });

    await new Promise((resolve, reject) => {
      this.socket.onopen = resolve;
      setTimeout(() => reject(new Error("voice socket timeout")), 8000);
    });

    await this.startMic();
    await this.startPlayback();
  }

  async startMic() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: SAMPLE_RATE,
        // Left ON deliberately: a training room has background noise, and
        // echo cancellation also stops the counterpart's own voice from
        // the speakers being picked up and detected as a barge-in.
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    this.micContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
    await this.micContext.audioWorklet.addModule(URL.createObjectURL(blob));

    const source = this.micContext.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(this.micContext, "capture");

    node.port.onmessage = (event) => {
      if (this.muted) return;
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.socket.send(event.data.buffer);
      }
    };

    source.connect(node);
    // Connected to the destination with zero gain: some browsers suspend
    // a worklet that has no downstream path, which would silently stop
    // capture.
    const silent = this.micContext.createGain();
    silent.gain.value = 0;
    node.connect(silent).connect(this.micContext.destination);
  }

  async rebuildPlayback() {
    if (this.playContext?.sampleRate === this.outputRate) return;
    await this.playContext?.close().catch(() => {});
    this.playContext = null;
    await this.startPlayback();
  }

  async startPlayback() {
    this.playContext = new AudioContext({ sampleRate: this.outputRate });
    // Autoplay policy: a context created before a user gesture starts
    // suspended, and audio would silently never play.
    if (this.playContext.state === "suspended") {
      await this.playContext.resume();
    }
    this.playCursor = this.playContext.currentTime;
  }

  enqueueAudio(arrayBuffer) {
    // Guard every precondition: Web Audio THROWS on a zero-length or
    // odd-length buffer, and an uncaught throw here kills playback for
    // the rest of the turn rather than just dropping one frame.
    if (!this.playContext) return;
    if (arrayBuffer.byteLength < 4 || arrayBuffer.byteLength % 2 !== 0) return;
    if (this.playContext.state === "suspended") {
      // Autoplay policy: a context can be suspended at any point, not
      // only at creation. Resuming is async, so the frame is dropped --
      // better one gap than a dead session.
      this.playContext.resume().catch(() => {});
      return;
    }

    const pcm = new Int16Array(arrayBuffer);
    const buffer = this.playContext.createBuffer(1, pcm.length, this.outputRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 0x8000;

    const source = this.playContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.playContext.destination);


    // Never schedule in the past: if the stream stalled, the cursor may
    // have fallen behind the clock, and starting at a past time plays the
    // chunk truncated or not at all.
    const startAt = Math.max(this.playCursor, this.playContext.currentTime + 0.02);
    source.start(startAt);
    this.playCursor = startAt + buffer.duration;

    this.scheduled.add(source);
    source.onended = () => this.scheduled.delete(source);
  }

  safeEnqueue(arrayBuffer) {
    try {
      this.enqueueAudio(arrayBuffer);
    } catch (err) {
      // Never let a malformed frame end the session. Reported once so a
      // real decoding problem is still visible rather than silently
      // swallowed on every frame.
      if (!this._audioErrorLogged) {
        this._audioErrorLogged = true;
        this.onEvent({ type: "audio_error", message: String(err) });
      }
    }
  }

  flushPlayback() {
    // Stop every scheduled buffer, including ones not yet started. This is
    // what makes interruption real rather than cosmetic.
    for (const source of this.scheduled) {
      try { source.stop(); } catch { /* already finished */ }
    }
    this.scheduled.clear();
    if (this.playContext) this.playCursor = this.playContext.currentTime;
  }

  interrupt() {
    this.flushPlayback();
    this.send({ type: "interrupt" });
  }

  sendText(text) {
    this.send({ type: "text", text });
  }

  send(message) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  setMuted(muted) {
    this.muted = muted;
  }

  async close() {
    this.send({ type: "stop" });
    this.flushPlayback();
    this.stream?.getTracks().forEach((track) => track.stop());
    await this.micContext?.close().catch(() => {});
    await this.playContext?.close().catch(() => {});
    this.socket?.close();
  }
}
