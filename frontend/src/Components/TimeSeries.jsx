import { useEffect, useState } from 'react';
import {
  LineChart, Line, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Brush
} from 'recharts';
import { getStationById } from './service';
import Loading from './Loading';
import StationDetails from './StationDetails';
import StationFilter from './StationFilter';
import { io } from "socket.io-client";

const socket = io("http://localhost:5000");
// Format timeseries data into unified [{ index, HE, HN, HZ }] format
function formatTimeSeries(timeseries, type = 'velocity') {
  const he = timeseries?.HE?.[type] || [];
  const hn = timeseries?.HN?.[type] || [];
  const hz = timeseries?.HZ?.[type] || [];

  const length = Math.min(he.length, hn.length, hz.length);

  return Array.from({ length }, (_, i) => ({
    index: i,
    HE: he[i],
    HN: hn[i],
    HZ: hz[i]
  }));
}

function TimeSeries({ stationId }) {
  const [station, setStation] = useState({});
  const [formattedData, setFormattedData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [zoomRange, setZoomRange] = useState([0, 0]);

  const [type, setType] = useState('acceleration')
  const [sensors, setSensors] = useState(['HE', 'HN', 'HZ'])
  const [showFilter, setShowFilter] = useState(false)



  // Initial load
  useEffect(() => {
    setLoading(true);
    getStationById(stationId)
      .then(res => {
        const st = res.data;
        const data = formatTimeSeries(st.time_series, type);
        setStation(st);
        setFormattedData(data);
        setZoomRange([0, data.length - 1]);
      })
      .catch(err => {
        console.error("Error loading timeseries:", err);
      })
      .finally(() => setLoading(false));
  }, [stationId, type,sensors]);


  // Real-time WebSocket updates
  useEffect(() => {
    const handleUpdate = (payload) => { 
      const relevant = payload?.data?.filter(
        rec => rec.station_id === stationId && sensors.includes(rec.component)
      );

      if (!relevant.length) return;

      const newPoint = { index: formattedData.length };
      sensors.forEach(s => newPoint[s] = null);

      for (const rec of relevant) {
        if (rec.component && type in rec) {
          newPoint[rec.component] = rec[type];
        }
      }

      setFormattedData(prev => [...prev, newPoint]);
      setZoomRange(prev => [Math.max(0, prev[0]), formattedData.length]);
    };

    socket.on("seismic_update", handleUpdate);
    return () => socket.off("seismic_update", handleUpdate);
  }, [stationId, type, sensors, formattedData]);


  return (
    <div className="TimeSeries container-fluid">
      {loading ? (
        <Loading />
      ) : (
        <div className="row">
          {/* Station Info */}
          <div className=" station_infos">
            <div>

              <h5 className='d-inline-block'> {showFilter ? "Filter By" : "Station Details"}  </h5>

              <button className='float-end  filter-btn  ' onClick={() => setShowFilter(!showFilter)}>  {!showFilter ? <><i class="fas fa-filter    "></i> Filter</> : <><i class="fas fa-chevron-left"></i> Back</>}  </button>
            </div>
            {showFilter ? <StationFilter selectedOption={type} onOptionChange={setType} selectedSensors={sensors} setSelectedSensors={setSensors} /> : <StationDetails station={station} />}

          </div>

          {/* Chart */}
          <div className="col">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={formattedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <YAxis />
                <Tooltip />
                <Legend />
                {sensors.includes("HE") && ( <Line type="monotone" dataKey="HE" stroke="#FF0000" name="HE" dot={false} /> )}
                {sensors.includes("HN") && ( <Line type="monotone" dataKey="HN" stroke="#007BFF" name="HN" dot={false} /> )}
                {sensors.includes("HZ") && (  <Line type="monotone" dataKey="HZ" stroke="#28A745" name="HZ" dot={false} /> )}
                <Brush
                  dataKey="index"
                  height={20}
                  startIndex={zoomRange[0]}
                  endIndex={zoomRange[1]}
                  onChange={(range) => {
                    if (range && typeof range.startIndex === 'number' && typeof range.endIndex === 'number') {
                      setZoomRange([range.startIndex, range.endIndex]);
                    }
                  }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

export default TimeSeries;
