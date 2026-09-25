import { useState, useEffect } from 'react';
import styles from './Admin.module.css';

// Hidden admin page: only reachable by typing /admin into the address bar.
// It is intentionally not linked from the navbar. The API key is kept in
// memory only and is gone after a reload.
function Admin() {
    const apiUrl = "https://api.marvinkrausser.com";

    const [apiKey, setApiKey] = useState("");
    const [limit, setLimit] = useState(100);
    const [offset, setOffset] = useState(0);
    const [loading, setLoading] = useState(false);
    const [confirmDelete, setConfirmDelete] = useState(false);
    const [response, setResponse] = useState(null);

    // Keep search engines away from this page.
    useEffect(() => {
        const meta = document.createElement("meta");
        meta.name = "robots";
        meta.content = "noindex, nofollow";
        document.head.appendChild(meta);
        return () => meta.remove();
    }, []);

    const request = async (label, method, path) => {
        if (!apiKey) {
            setResponse({ label, status: "No API key entered", body: null, ok: false });
            return;
        }

        setLoading(true);
        setConfirmDelete(false);

        try {
            const res = await fetch(`${apiUrl}${path}`, {
                method,
                headers: { "Authorization": `Bearer ${apiKey}` },
            });

            const text = await res.text();
            let body;
            try {
                body = JSON.parse(text);
            } catch {
                body = text;
            }

            setResponse({ label, status: `${res.status} ${res.statusText}`, body, ok: res.ok });
        } catch (e) {
            setResponse({ label, status: "Network error", body: String(e), ok: false });
        } finally {
            setLoading(false);
        }
    };

    const getReviews = () =>
        request("Get reviews", "GET", `/review?limit=${limit}&offset=${offset}`);

    const deleteAll = () => {
        // Two-step confirmation instead of window.confirm.
        if (!confirmDelete) {
            setConfirmDelete(true);
            return;
        }
        request("Delete all reviews", "DELETE", "/review");
    };

    const reviews = Array.isArray(response?.body?.data) ? response.body.data : null;

    return (
        <div className='site-box'>
            <h1 className='site-headline'>Admin</h1>

            <div className={styles.panel}>
                <label className={styles.field}>
                    <span>API key</span>
                    <input
                        type="password"
                        autoComplete="off"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value.trim())}
                        placeholder="Enter API key"
                    />
                </label>

                <div className={styles.row}>
                    <label className={styles.field}>
                        <span>Limit</span>
                        <input
                            type="number"
                            min="1"
                            max="500"
                            value={limit}
                            onChange={(e) => setLimit(Math.min(500, Math.max(1, Number(e.target.value) || 1)))}
                        />
                    </label>
                    <label className={styles.field}>
                        <span>Offset</span>
                        <input
                            type="number"
                            min="0"
                            value={offset}
                            onChange={(e) => setOffset(Math.max(0, Number(e.target.value) || 0))}
                        />
                    </label>
                </div>

                <div className={styles.buttons}>
                    <button className='custom-button' onClick={getReviews} disabled={loading}>
                        Get reviews
                    </button>
                    <button
                        className={`custom-button ${styles.danger}`}
                        onClick={deleteAll}
                        onBlur={() => setConfirmDelete(false)}
                        disabled={loading}
                    >
                        {confirmDelete ? "Click again to delete ALL" : "Delete all reviews"}
                    </button>
                </div>
            </div>

            <div className={styles.output}>
                <div className={styles["output-head"]}>
                    <span>Response</span>
                    {loading && <span className={styles.dim}>Loading…</span>}
                    {!loading && response && (
                        <span className={response.ok ? styles.ok : styles.fail}>
                            {response.label}: {response.status}
                        </span>
                    )}
                </div>

                {reviews && reviews.length > 0 && (
                    <div className={styles["table-wrap"]}>
                        <table className={styles.table}>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Created</th>
                                    <th>Website</th>
                                    <th>Rating</th>
                                    <th>Text</th>
                                </tr>
                            </thead>
                            <tbody>
                                {reviews.map((r) => (
                                    <tr key={r.id}>
                                        <td>{r.id}</td>
                                        <td>{r.created_at ? new Date(r.created_at).toLocaleString() : ""}</td>
                                        <td>{r.website}</td>
                                        <td>{r.rating}</td>
                                        <td className={styles.text}>{r.text}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {reviews && reviews.length === 0 && <p className={styles.dim}>No reviews.</p>}

                {response && (
                    <pre className={styles.raw}>
                        {typeof response.body === "string"
                            ? response.body
                            : JSON.stringify(response.body, null, 2)}
                    </pre>
                )}

                {!response && <p className={styles.dim}>No request sent yet.</p>}
            </div>
        </div>
    );
}

export default Admin;
