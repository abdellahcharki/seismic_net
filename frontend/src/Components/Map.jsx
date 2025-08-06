import { useEffect, useState } from 'react';
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  Polyline,
  useMapEvent
} from 'react-leaflet';
import { getAllLinks, getAllStations } from "./service";
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { getRiskColor } from "./service";
import { io } from "socket.io-client";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const socket = io("http://localhost:5000");

// إصلاح أيقونات Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const Map = ({ selectStation, selectedStation }) => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

  // جلب البيانات عند أول تحميل
  useEffect(() => {
    const fetchData = async () => {
      const [stationsRes, linksRes] = await Promise.all([
        getAllStations(),
        getAllLinks()
      ]);
      setGraphData({
        nodes: stationsRes.data,
        links: linksRes.data
      });
    };
    fetchData();
  }, []);

  // البحث عن محطة بالمعرف
  const getStationById = (id) =>
    graphData.nodes.find(station => station.id === id);

  const tesk_msg = (stab,risk)=>{
    if(stab == 'unstable') return "unstable"
    else if (stab == 'stable' &&  risk == 'high' ) return 'High Risk'
    else if (stab == 'stable' &&  risk == 'low' ) return 'Stable'
  }

  // استقبال التحديثات اللحظية
  useEffect(() => {
    const handleStationChange = (payload) => {
      console.log("Received station update:", payload);
      toast.info(`Station ${payload.station_id} is now ${tesk_msg(payload.status,payload.risk)}`);

      setGraphData(prev => {
        const updatedNodes = prev.nodes.map(node =>
          node.id === payload.station_id
            ? { ...node, stability: payload.status, risk: payload.risk }
            : node
        );
        return {
          ...prev,
          nodes: updatedNodes
        };
      });
    };

    socket.on("station_update", handleStationChange);
    return () => {
      socket.off("station_update", handleStationChange);
    };
  }, []);

  // النقر على الخلفية لإلغاء التحديد
  const BackgroundClickHandler = ({ onMapClick }) => {
    useMapEvent('click', () => {
      onMapClick();
    });
    return null;
  };

  return (
    <>
      <ToastContainer position="top-right" autoClose={3000} />
      <MapContainer
        center={[35.5858, -39.5121]}
        zoom={3}
        style={{ height: '100%', width: '100%' }}
      >
        <BackgroundClickHandler onMapClick={() => selectStation(null)} />
        <TileLayer
          attribution='© OpenStreetMap contributors'
          url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
        />

        {/* رسم العقد (المحطات) */}
        {graphData.nodes.map((station, i) => (
          <CircleMarker
            key={i}
            center={[station.latitude, station.longitude]}
            radius={10}
            pathOptions={{
              color: station.id === selectedStation ? 'blue' : getRiskColor(station.risk),
              weight: station.id === selectedStation ? 3 : 1,
              fillColor: getRiskColor(station.risk, station.stability),
              fillOpacity: 0.8,
            }}
            eventHandlers={{
              click: () => selectStation(station.id)
            }}
          >
            <Popup>
              <b>{station.id}</b><br />
            </Popup>
          </CircleMarker>
        ))}

        {/* رسم الروابط */}
        {graphData.links.map((link, i) => {
          const source = getStationById(link.source);
          const target = getStationById(link.target);
          if (source && target) {
            return (
              <Polyline
                key={i}
                positions={[
                  [source.latitude, source.longitude],
                  [target.latitude, target.longitude],
                ]}
                color="#6e6e6e"
                weight={1}
                opacity={0.4}
              />
            );
          }
          return null;
        })}
      </MapContainer>
    </>
  );
};

export default Map;