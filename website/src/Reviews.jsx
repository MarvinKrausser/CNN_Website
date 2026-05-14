import { useState, useRef, useEffect } from 'react';
import './Reviews.css';

function Reviews() {
    const [error, setError] = useState(null);
    const [selectedWebsite, setSelectedWebsite] = useState("Not Specified");
    const [selectWebsiteVisibility, setSelectWebsiteVisibility] = useState(false);
    const websiteSelectButton = useRef();
    const [text, setText] = useState("");


    const fetchReviews = async () => {
        const response = await fetch(`api.marvinkrausser/review`, {
            method: "GET",
        });

        if (!response.ok) {
            setError(true);
            return;
        }
        else {
            setError(false);
        }

        const result = await response.json();
    }

    const sendReview = async (review) => {

        const response = await fetch(`api.marvinkrausser/review`, {
            method: "POST",
            body: JSON.stringify(review),
        });

        if (!response.ok) {
            setError(true);
            return;
        }
        else {
            setError(false);
        }

        const result = await response.json();
    }

    const changeSelectedWebsite = (website) => {
        setSelectedWebsite(website);
    };

    const websites = [
        "CNN (current)",
        "Portfolio",
        "Not Specified"
    ];

    useEffect(() => {
        function handleClick(event) {
            if (websiteSelectButton.current && !websiteSelectButton.current.contains(event.target)
            ) {
                setSelectWebsiteVisibility(false);
            }
        }

        document.addEventListener("click", handleClick);

        return () => {
            document.removeEventListener("click", handleClick);
        };
    }, []);


    return (<>
        <div className='site-box'>
            <h1 className='site-headline'>Reviews</h1>

            <div className='input-box'>
                <div className='review-select-dropdown'>
                    <button ref={websiteSelectButton} className={selectWebsiteVisibility ? "review-select-button active" : "review-select-button"} onClick={() => { setSelectWebsiteVisibility(!selectWebsiteVisibility) }}>
                        {selectedWebsite}
                    </button>
                    {selectWebsiteVisibility && <div className='review-select-menu'>
                        {websites.map((site, i) =>
                            selectedWebsite !== site && (
                                <div
                                    key={site}
                                    className={i >= (websites.length - (selectedWebsite == websites.at(-1) ? 2 : 1)) ? "review-select-item last" : "review-select-item"}
                                    onClick={() => changeSelectedWebsite(site)}
                                >
                                    {site}
                                </div>
                            )
                        )}
                    </div>}
                </div>

                <input
                    type="text"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Type something..."
                />
            </div>
        </div>
    </>)
}

export default Reviews;