import styles from './Object_Detection.module.css';
import { useRef, useEffect } from 'react'

function Object_Detection() {
    const videoRef = useRef(null);

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

    const captureImage = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        const width = video.videoWidth;
        const height = video.videoHeight;

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");

        ctx.drawImage(video, 0, 0, width, height);

        const dataUrl = canvas.toDataURL("image/png");

        setImage(dataUrl);
    };


    return (
        <div className='site-box'>
            <h1 className='site-headline'>Face Detection</h1>
            <div id={styles.container}>
                <video autoPlay={true} id={styles.videoElement} ref={videoRef}>

                </video>
            </div>
        </div>
    )
}

export default Object_Detection;