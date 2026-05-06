import { useState } from 'react'
import { useRef } from "react";
import './Bird_CNN.css';

function Bird_CNN() {
    const apiUrl = process.env.NODE_ENV === "development"
        ? "https://api.marvinkrausser.com"
        : "https://api.marvinkrausser.com";

    const [file, setFile] = useState(null);
    const [birdClass, setBirdClass] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [error, setError] = useState(false);
    const [preview, setPreview] = useState("/cherry_bird.jpeg");
    const [loading, setLoading] = useState(false);
    const fileInputRef = useRef(null);
    const uploadButton = useRef(null);
    const scrollRefClassifiction = useRef(null);
    const scrollRefUploadButton = useRef(null);

    const isPartiallyInViewport = (el) => {
        if (!el) return false;

        const rect = el.getBoundingClientRect();

        return (
            rect.top < window.innerHeight &&
            rect.bottom > 0
        );
    };

    const handleImageDivClick = () => {
        fileInputRef.current.click();
    }

    const handleImage = (e) => {
        if (!e.target.files[0]) { return; }
        setFile(e.target.files[0]);
        setPreview(URL.createObjectURL(e.target.files[0]));
        uploadButton.current.classList.remove("inactive");

        scrollRefUploadButton.current.scrollIntoView({ behavior: "smooth" });

        setConfidence(null);
        setBirdClass(null);

    };

    const sendImage = async (e) => {
        if (!file) return;

        scrollRefClassifiction.current.scrollIntoView({ behavior: "smooth" });

        const formData = new FormData();
        formData.append("file", file);

        setLoading(true);

        try {
            const response = await fetch(`${apiUrl}/predict`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                setError(true);
                return;
            }
            else {
                setError(false);
            }

            const result = await response.json();

            setBirdClass(result["class"]);
            const confidence = result["confidence"];
            setConfidence(`${Math.round(confidence * 100)}%`);
        } catch (e) {
            setError(true);
        }
        finally {
            setLoading(false);
        }
    };

    return (
        <>
            <div className='site-box'>
                <h1 id='site-headline'>Bird Species Expert</h1>
                <div className='content-box' style={{ display: "flex", flexWrap: "wrap", justifyContent: "center" }}>
                    <div className='explanation-box left'>
                        <h2 style={{ color: "rgb(47, 168, 208)" }}>Explanation</h2>
                        <span>Select an image and upload it to our bird expert. You will receive a classification and how certain the expert is with her opinion. Be aware that the expert may not be always right.</span>
                    </div>

                    <div className='request-box'>
                        <div ref={scrollRefUploadButton} className='input-box'>
                            <div className='button-div'>
                                <input disabled={loading} ref={fileInputRef} type="file" id='fileUpload' accept="image" onChange={handleImage} style={{ display: "none" }} />
                                <label htmlFor="fileUpload" className="custom-button">
                                    Select Image
                                </label>
                            </div>
                            <div className='button-div'>
                                <button id='button-send' onClick={sendImage} style={{ display: "none" }} disabled={loading} />
                                <label htmlFor="button-send" className="custom-button inactive" ref={uploadButton}>
                                    Ask Expert
                                </label>
                            </div>

                        </div>

                        <div className='image-box'>
                            <img
                                onClick={handleImageDivClick}
                                disabled={loading}
                                src={preview}
                                alt="preview"
                            />
                        </div>

                        <div ref={scrollRefClassifiction} className='loader-container'>
                            {loading && <div className='loader'></div>}
                        </div>


                        <div className='response-block'>
                            <div className='content-block class'>
                                <h3 className='conten-block-text'>Bird Species: </h3>
                                {birdClass && <p id='bird-class-text' className='conten-block-text'>{birdClass}</p>}
                            </div>
                            <div className='content-block confidence'>
                                <h4 className='conten-block-text'>Model Confidence: </h4>
                                {confidence && <p id='bird-confidence-text' className='conten-block-text'>{confidence}</p>}
                            </div>
                            {error && <h4 className='conten-block-text'>An Error has uccured. Please try again later.</h4>}
                        </div>
                    </div>

                    <div className='explanation-box right'>
                        <h3>Model Architecture</h3>
                        <div style={{ display: "inline" }}>
                            <span>The model used for classification is a convolutional neural network. The model uses who knows how many layers, a dropout and multiple batchnormalsation.</span>
                        </div>
                    </div>
                </div>
            </div >
        </>
    );
}

export default Bird_CNN;