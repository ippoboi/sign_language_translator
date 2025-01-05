"use client";
import { Book, Camera, Play, RotateCcw, Square } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useWebSocket } from "../hook/useWebSocket";
import { Video } from "lucide-react";

export default function Home() {
  const {
    videoRef,
    canvasRef,
    status,
    fps,
    latency,
    isHandDetected,
    startStream,
    stopStream,
    isStreaming,
    logs,
  } = useWebSocket();

  return (
    <div className="min-h-screen p-6 font-mono bg-slate-100 space-y-6">
      {/* Header */}
      <header className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <Image
            src="/sign-lang-logo.svg"
            alt="SignLang"
            width={32}
            height={32}
          />
          <span className="text-xl font-medium">SignLang</span>
        </div>
        <div className="flex items-center gap-4">
          <Link
            href="https://www.signasl.org/"
            className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-gray-100"
            target="_blank"
          >
            <Book className="w-5 h-5" strokeWidth={1.5} />
            Sign dictionary
          </Link>
          {/* <button className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-gray-100">
            <Settings className="w-5 h-5" strokeWidth={1.5} />
            Settings
          </button> */}
        </div>
      </header>

      {/* Main Content */}
      <main className="grid grid-cols-5 gap-6">
        {/* Video Section */}
        <div className="flex-1 bg-slate-200 col-span-3 rounded-xl aspect-video relative overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-full object-cover"
          />
          <canvas ref={canvasRef} className="hidden" />

          <div className="absolute bottom-4 bg-white p-1 rounded-xl left-1/2 -translate-x-1/2 flex gap-1 shadow-lg">
            <button className="p-3 bg-slate-100 rounded-lg">
              <Video
                className="w-6 h-6"
                strokeWidth={1.5}
                stroke="none"
                fill="black"
              />
            </button>
            {!isStreaming ? (
              <button
                className="p-3 bg-blue-600 rounded-lg"
                onClick={startStream}
              >
                <Play
                  className="w-6 h-6 text-white"
                  stroke="none"
                  fill="white"
                />
              </button>
            ) : (
              <button
                className="p-3 bg-red-600 rounded-lg"
                onClick={stopStream}
              >
                <Square
                  className="w-6 h-6 text-white"
                  stroke="none"
                  fill="white"
                />
              </button>
            )}
            <button className="p-3 bg-slate-100 rounded-lg">
              <RotateCcw className="w-5 h-5" strokeWidth={1.5} />
            </button>
          </div>
        </div>

        {/* Right column */}
        <div className="w-full space-y-6 col-span-2 h-full flex flex-col">
          {/* Status box */}
          <div className="bg-white rounded-xl p-5 shadow-sm">
            <div className="flex gap-12">
              <div>
                <div className="text-sm text-gray-600">Status</div>
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      status === "connected" ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <span
                    className={
                      status === "connected" ? "text-green-500" : "text-red-500"
                    }
                  >
                    {status === "connected" ? "Connected" : "Disconnected"}
                  </span>
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600">FPS</div>
                <div>{Math.round(fps)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Latency</div>
                <div>{latency}ms</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Detection</div>
                <div>{isHandDetected ? "Yes" : "No"}</div>
              </div>
            </div>
          </div>

          {/* Translations box - Remove h-full, add flex-1 */}
          <div className="bg-white rounded-xl p-4 gap-4 shadow-sm flex-1 flex flex-col">
            <div className="flex justify-between items-center">
              <h2 className="text-sm font-medium">Translations box</h2>
              <span className="text-sm text-slate-600">97.8% accuracy</span>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 text-gray-400 flex-1">
              Translations will appear here...
            </div>
          </div>
        </div>
      </main>

      <div className="bg-white rounded-xl p-4 shadow-sm flex flex-col min-h-[400px]">
        <h2 className="text-sm font-medium mb-3">Console logging</h2>
        <div className="bg-gray-50 rounded-lg p-4 text-sm font-mono flex-1 overflow-y-auto">
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <div key={index} className="text-gray-600">
                {log}
              </div>
            ))
          ) : (
            <div className="text-gray-400">
              Detection details will appear here...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
