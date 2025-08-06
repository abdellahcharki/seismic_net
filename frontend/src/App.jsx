

import NavBar from "./Components/NavBar.jsx";
import { useEffect, useState } from "react";
import TimeSeries from "./Components/TimeSeries.jsx";
import Graph from "./Components/Graph.jsx";
import Map from "./Components/Map.jsx";
 import { io } from "socket.io-client";
const socket = io("http://localhost:5000");

function App() {
  const [showStat, setShowStat] = useState((localStorage.getItem('ismap') || false));
  const [selectedStationID, setSelectedStationID] = useState("")
 
 

  return (
    <div className="app">
      <NavBar show={showStat} setShow={setShowStat} />
      <div className='viewSpace'>
        {showStat ? <Graph   selectStation={setSelectedStationID} selectedStation={selectedStationID} /> : 
        <Map   selectStation={setSelectedStationID} selectedStation={selectedStationID} />}
      </div>
      {selectedStationID ? <TimeSeries stationId={selectedStationID} /> : null}

    </div>
  );
}
export default App;
