import styles from './Object_Detection.module.css';
import { useRef, useEffect, useState } from 'react'

function Object_Detection() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(false);
    const [bboxes, setBboxes] = useState(null)
    const socketRef = useRef(null);
    const ctxRef = useRef(null);
    const captureIntervalRef = useRef(null);
    const drawIntervalRef = useRef(null);
    const [drawing, setDrawing] = useState(null);
    const [sending, setSending] = useState(null);

    useEffect(() => {
        if (canvasRef.current) {
            ctxRef.current = canvasRef.current.getContext("2d");
        }
    }, []);

    useEffect(() => {
        async function startWebcam() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false,
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err) {
                console.error("Error accessing webcam:", err);
            }
        }

        startWebcam();

        return () => {
            if (videoRef.current?.srcObject) {
                const tracks = videoRef.current.srcObject.getTracks();
                tracks.forEach((track) => track.stop());
            }
        };
    }, []);

    const drawImage = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        const width = video.videoWidth;
        const height = video.videoHeight;

        const canvasW = canvas.width;
        const canvasH = canvas.height;

        const ctx = ctxRef.current;
        if (!ctx) return;

        ctx.drawImage(video, 0, 0, canvasW, canvasH);

        ctx.strokeStyle = "red";
        ctx.lineWidth = 4;

        if (bboxes) {
            bboxes.forEach(edge => {
                ctx.strokeRect(edge[0], edge[1], edge[2] - edge[0], edge[3] - edge[1]);
            });
        }
    };

    const sendImage = () => {
        const canvas = canvasRef.current;

        canvas.toBlob(async (blob) => {
            if (!blob) return;

            setLoading(true)

            try {
                socketRef.current?.send(blob)
            } catch (e) {
                setError(true);
            }
            finally {
                setLoading(false);
            }
        }, "image/jpeg", 0.9);
    };

    const startRecording = () => {
        const socket = new WebSocket("wss://api.marvinkrausser.com/predict_face");
        socketRef.current = socket;

        socket.onopen = () => {
            console.log("Connected");
            setDrawing(setInterval(drawImage, 1000 / 60));
            setSending(setInterval(sendImage, 100));
        }
        socket.onmessage = (event) => {
            setBboxes(JSON.parse(event.data).bboxes);
        }
        socket.onerror = console.error;
        socket.onclose = () => console.log("Disconnected");
    }

    return (
        <div className='site-box'>
            <h1 className='site-headline'>Face Detection</h1>
            <div id={styles.container}>
                <video autoPlay={true} ref={videoRef} style={{ display: "none" }}></video>
                <canvas ref={canvasRef} width={1000} height={800} />
            </div>
            <button onClick={startRecording}>Click</button>
        </div>
    )
}

export default Object_Detection;