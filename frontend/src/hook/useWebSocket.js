import { useState, useEffect, useRef } from "react";

export function useWebSocket() {
  const [status, setStatus] = useState("disconnected");
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [isHandDetected, setIsHandDetected] = useState(false);
  const [landmarks, setLandmarks] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [logs, setLogs] = useState([]);

  const wsRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const isRunningRef = useRef(false);
  const lastPingTimeRef = useRef(0);
  const lastFrameTimeRef = useRef(0);

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          frameRate: { ideal: 15, max: 30 },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      connectWebSocket();
      setIsStreaming(true);
    } catch (error) {
      console.error("Error accessing camera:", error);
    }
  };

  const connectWebSocket = () => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws");

    wsRef.current.onopen = () => {
      setStatus("connected");
      isRunningRef.current = true;
      requestAnimationFrame(sendFrames);
    };

    wsRef.current.onclose = () => {
      setStatus("disconnected");
      isRunningRef.current = false;
    };

    wsRef.current.onmessage = (event) => {
      const response = JSON.parse(event.data);
      const now = performance.now();

      setFps(response.fps || 0);
      if (lastPingTimeRef.current) {
        setLatency(Math.round(now - lastPingTimeRef.current));
      }

      setIsHandDetected(response.detected);
      if (response.landmarks) {
        setLandmarks(response.landmarks);
        if (response.detected) {
          addLog(`Hands detected - ${response.landmarks.length} hand(s)`);
        }
      }
    };
  };

  const sendFrames = (timestamp) => {
    if (!isRunningRef.current) return;

    // Throttle to ~15 FPS
    if (timestamp - lastFrameTimeRef.current < 66) {
      requestAnimationFrame(sendFrames);
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (canvas && video && wsRef.current?.readyState === WebSocket.OPEN) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      const frame = canvas.toDataURL("image/jpeg", 0.7);
      lastPingTimeRef.current = performance.now();
      wsRef.current.send(frame);
    }

    lastFrameTimeRef.current = timestamp;
    requestAnimationFrame(sendFrames);
  };

  const stopStream = () => {
    isRunningRef.current = false;
    if (wsRef.current) {
      wsRef.current.close();
    }
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
    }
    setIsStreaming(false);
  };

  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prevLogs) => {
      const newLogs = [`[${timestamp}] ${message}`, ...prevLogs];
      // Keep only the last 50 logs
      return newLogs.slice(0, 50);
    });
  };

  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  return {
    videoRef,
    canvasRef,
    status,
    fps,
    latency,
    isHandDetected,
    landmarks,
    startStream,
    stopStream,
    isStreaming,
    logs,
    addLog,
  };
}
