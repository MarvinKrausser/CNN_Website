import { Routes, Route, Navigate } from "react-router-dom";
import Menu from './Menu'
import Homepage from './Homepage';
import Bird_CNN from './Bird_CNN';
import Reviews from './Reviews.jsx';
import Object_Detection from './Object_Detection.jsx';
import Admin from './Admin.jsx';

function App() {

  return (
    <>
      <Menu />
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/bird_cnn" element={<Bird_CNN />} />
        <Route path="/object_detection" element={<Object_Detection />} />
        <Route path="/reviews" element={<Reviews />} />
        {/* Not linked in the navbar on purpose: reachable only via the URL. */}
        <Route path="/admin" element={<Admin />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default App
