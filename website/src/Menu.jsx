import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import './Menu.css'

const links = [
    { to: "/", label: "Home" },
    { to: "/bird_cnn", label: "Birds" },
    { to: "/object_detection", label: "YOLO" },
    { to: "/reviews", label: "Reviews" },
];

function Menu() {
    const [open, setOpen] = useState(false);

    const linkClass = ({ isActive }) =>
        `navbar-item${isActive ? " active" : ""}`;

    return (
        <nav id="navbar-main">
            <div id="navbar-inner">
                <NavLink to="/" className="navbar-brand" onClick={() => setOpen(false)}>
                    Marvin Krausser
                </NavLink>

                <button
                    id="navbar-toggle"
                    aria-label={open ? "Close menu" : "Open menu"}
                    aria-expanded={open}
                    onClick={() => setOpen(!open)}
                >
                    <span className={open ? "bar bar-1 open" : "bar bar-1"} />
                    <span className={open ? "bar bar-2 open" : "bar bar-2"} />
                    <span className={open ? "bar bar-3 open" : "bar bar-3"} />
                </button>

                <div id="navbar-links" className={open ? "open" : ""}>
                    {links.map(({ to, label }) => (
                        <NavLink
                            key={to}
                            end={to === "/"}
                            className={linkClass}
                            to={to}
                            onClick={() => setOpen(false)}
                        >
                            {label}
                        </NavLink>
                    ))}
                </div>
            </div>
        </nav>
    );
}

export default Menu;
