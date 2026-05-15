import { useState, useRef, useEffect } from 'react';
import './Reviews.css';
import SelectDropdown from './components/SelectDropdown';

function Reviews() {
    const apiUrl = process.env.NODE_ENV === "development"
        ? "https://api.marvinkrausser.com"
        : "https://api.marvinkrausser.com";

    const [error, setError] = useState(null);
    const [selectedWebsite, setSelectedWebsite] = useState("Not Specified");
    const [selectedRating, setSelectedRating] = useState(1);
    const [text, setText] = useState("");
    const [loading, setLoading] = useState(false);
    const [sendSuccess, setSendSuccess] = useState(false);

    const MAX_TEXT = 900;


    const fetchReviews = async () => {
        const response = await fetch(`${apiUrl}/review`, {
            method: "GET",
            headers: {
                "authorization": `Bearer 1234`,
            },
        });

        if (!response.ok) {
            setError(true);
            return;
        }
        else {
            setError(false);
        }

        const result = await response.json();

        console.log(result);
    }

    const sendReview = async () => {
        const review = {
            "website": selectedWebsite,
            "rating": selectedRating,
            "text": text,
            "date": "today"
        }

        console.log(JSON.stringify(review));

        setLoading(true);
        setError(false);
        setSendSuccess(false);

        try {
            const response = await fetch(`${apiUrl}/review`, {
                method: "POST",
                body: JSON.stringify(review),
                headers: {
                    "Content-Type": "application/json"
                }
            });

            if (!response.ok) {
                setError(true);
                return;
            }
            else {
                setError(false);
            }

            setSendSuccess(true);
            const result = await response.json();
        } catch (e) {
            setError(true);
        }
        finally {
            setLoading(false);
        }
    }

    const changeSelectedWebsite = (website) => {
        setSelectedWebsite(website);
    };

    const changeSelectedRating = (rating) => {
        setSelectedRating(rating);
    }

    const websites = [
        "CNN (current)",
        "Portfolio",
        "Not Specified"
    ];

    const ratings = [1, 2, 3, 4, 5];

    const handleTextInput = (e) => {
        const textInput = e.target.value;

        if (textInput.length <= MAX_TEXT || textInput.length < text.length) {
            setText(textInput);
        }
    };

    return (<>
        <div className='site-box'>
            <h1 className='site-headline'>Reviews</h1>
            <p className='review-description'>I would greatly appreciate receiving a review. You are welcome to write it in either English or German. All fields are optional and all reviews are submitted anonymously.</p>



            <div className='input-box'>
                <div className='input-component-box'>
                    <p className='description'>Website</p>
                    <SelectDropdown
                        listClass="flex"
                        parentClass="input-component"
                        values={websites}
                        defaultValue={websites.at(-1)}
                        classNames={"website"}
                        changeInputParent={changeSelectedWebsite}
                    />
                </div>

                <div className='input-component-box'>
                    <p className='description'>Rating</p>
                    <SelectDropdown
                        listClass="grid"
                        parentClass="input-component"
                        values={ratings}
                        defaultValue={ratings.at(0)}
                        classNames={"rating"}
                        changeInputParent={changeSelectedRating}
                    />
                </div>

                <div className='input-component-box'>
                    <div className='button-div input-component'>
                        <button id='button-send' style={{ display: "none" }} onClick={sendReview} />
                        <label htmlFor="button-send" className="custom-button send-review">
                            Send Review
                        </label>

                        <div className='loader-box'>
                            {loading && <div className="loader2"></div>}
                            {sendSuccess && <p className='description second'>Review sent</p>}
                            {error && <p className='description second'>Error</p>}
                        </div>
                    </div>
                </div>

                <div className='input-component-box text'>
                    <p className='description'>Review</p>
                    <textarea
                        autoCorrect="off"
                        autoCapitalize="off"
                        spellCheck={false}
                        className='review-text input-component'
                        type="text"
                        value={text}
                        onChange={(e) => handleTextInput(e)}
                        placeholder="Share your Opinion"
                    />
                    <p className='description second'>{text.length}/{MAX_TEXT}</p>
                </div>
            </div>
        </div>
    </>)
}

export default Reviews;