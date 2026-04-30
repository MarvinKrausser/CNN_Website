import { useState } from 'react'
import { useRef } from "react";
import './Bird_CNN.css';

function Bird_CNN() {
    const apiUrl = process.env.NODE_ENV === "development"
        ? "http://134.60.154.7:8000"
        : "http://134.60.154.7:8000";

    const [file, setFile] = useState(null);
    const [birdClass, setBirdClass] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [error, setError] = useState(false);
    const [preview, setPreview] = useState("\\cherry_bird.jpeg");
    const [loading, setLoading] = useState(false);
    const fileInputRef = useRef(null);
    const uploadButton = useRef(null);
    const scrollRef = useRef(null);

    const handleImageDivClick = () => {
        fileInputRef.current.click();
    }

    const handleImage = (e) => {
        if (e.target.files[0]) {
            setFile(e.target.files[0]);
            setPreview(URL.createObjectURL(e.target.files[0]));
            uploadButton.current.classList.remove("inactive");
        }
        setConfidence(null);
        setBirdClass(null);
    };

    const sendImage = async (e) => {
        if (!file) return;

        scrollRef.current.scrollIntoView({ behavior: "smooth" });

        const formData = new FormData();
        formData.append("file", file);

        try {
            setLoading(true);
            const response = await fetch(`${apiUrl}/predict`, {
                method: "POST",
                body: formData,
            });

            setLoading(false);

            if (!response.ok) {
                setError(true);
            }
            else {
                setError(false);
            }

            const result = await response.json();
            setBirdClass(result["class"]);
            const confidence = result["confidence"];
            setConfidence(`${Math.round(confidence * 100)}%`);
        }
        catch (error) {
            console.error("Upload failed:", error);
        }
    };

    return (
        <>
            <div className='site-box' style={{ marginTop: "150px" }}>
                <h1 id='site-headline'>Bird Species Expert</h1>
                <div className='content-box' style={{ display: "flex", flexWrap: "wrap", justifyContent: "center" }}>
                    <div className='explanation-box left'>
                        <h2 style={{ color: "rgb(47, 168, 208)" }}>Explanation</h2>
                        <span>Select an image and upload it to our bird expert. You will receive a classification and how certain the expert is with her opinion. Be aware that the expert may not be always right.</span>
                    </div>

                    <div className='request-box' style={{ height: "800px", width: "700px", display: "flex", flexDirection: "column", alignItems: "center", flex: "0 0 auto", margin: "100px 50px 0px 50px" }}>
                        <div className='input-box' style={{ marginBottom: "80px", width: "500px", height: "40px", display: "flex", justifyContent: "space-evenly", alignItems: "center" }}>
                            <div>
                                <input disabled={loading} ref={fileInputRef} type="file" id='fileUpload' accept="image/jpeg" onChange={handleImage} style={{ display: "none" }} />
                                <label htmlFor="fileUpload" className="custom-button">
                                    Select Image
                                </label>
                            </div>
                            <div>
                                <button id='button-send' onClick={sendImage} style={{ display: "none" }} disabled={loading} />
                                <label htmlFor="button-send" className="custom-button inactive" ref={uploadButton}>
                                    Ask Expert
                                </label>
                            </div>

                        </div>

                        <div className='image-box' onClick={handleImageDivClick} disabled={loading}>
                            <img
                                src={preview}
                                alt="preview"
                                style={{ maxHeight: "400px", height: "auto", userSelect: "none", maxWidth: "700px" }}
                            />
                        </div>

                        <div ref={scrollRef} style={{ paddingTop: "10px", marginTop: "30px", width: "85px", height: "35px", display: "flex", justifyContent: "center", minHeight: "25px", minWidth: "85px" }}>
                            {loading && <div className='loader'></div>}
                        </div>


                        <div className='response-block' style={{ marginTop: "40px", width: "80%", margin: "40px", display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "10px" }}>
                            <div className='content-block class'>
                                <h3>Bird Species: </h3>
                                <p id='bird-class-text'>{birdClass}</p>
                            </div>
                            <div className='content-block confidence'>
                                <h4>Model Confidence: </h4>
                                <p id='bird-confidence-text'>{confidence}</p>
                            </div>
                            {error && <h4>An Error has uccured. Please try again later.</h4>}
                        </div>
                    </div>

                    <div className='explanation-box right'>
                        <h3>Model Architecture</h3>
                        <div style={{ display: "inline" }}>
                            <span>The model used for classification is a convolutional neural network (CNN) based on depthwise separable convolutions, as introduced in the </span>
                            <a href='https://arxiv.org/pdf/1610.02357' target='_blank' rel='noopener noreferrer'>Xception: Deep Learning with Depthwise Separable Convolutions</a>
                            <span> paper by François Chollet. It uses who knows how many layers, a dropout and multiple batchnormalsation.</span>
                        </div>
                    </div>
                </div>
            </div >
        </>
    );
}

export default Bird_CNN;