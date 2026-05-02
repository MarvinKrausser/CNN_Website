import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'
import Menu from './Menu'
import Homepage from './Homepage';
import Bird_CNN from './Bird_CNN';

function App() {

  return (
    <>
      <Menu />
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/bird_cnn" element={<Bird_CNN />} />
      </Routes>
    </>
  );
}

export default App
