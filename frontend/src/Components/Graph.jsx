import React, { useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import axios from 'axios';
import { io } from "socket.io-client";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
const socket = io("http://localhost:5000");
 

  const tesk_msg = (stab,risk)=>{
    if(stab == 'unstable') return "unstable"
    else if (stab == 'stable' &&  risk == 'high' ) return 'High Risk'
    else if (stab == 'stable' &&  risk == 'low' ) return 'Stable'
  }

    const getRiskColor = (risk,stab) => {
        if(stab=='unstable') return  "red";
         else if(stab=='stable' && risk=="high") return "orange";
        else if(stab=='stable' && risk=="low")  return "#3CB043";
        else return "gray"
    };

const Graph = ({selectStation,selectedStation}) => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const nodesRes = await axios.get('http://localhost:5000/hybrid/nodes');
        const linksRes = await axios.get('http://localhost:5000/hybrid/links');

        setGraphData({
          nodes: nodesRes.data,
          links: linksRes.data
        });
      } catch (err) {
        console.error('Error fetching graph data:', err);
      }
    };

    fetchData();
  }, []);


useEffect(() => {
  const handleStationChange = (payload) => {
    console.log("Received station update:", payload);   
    toast.info(`Station ${payload.station_id} is now ${tesk_msg(payload.status,payload.risk)}`);

    setGraphData(prev => {
      // ⚠️ تعديل مباشر للعقد بدلاً من map()
      const node = prev.nodes.find(n => n.id === payload.station_id);
      if (node) {
        node.stability = payload.status;
        node.risk = payload.risk;
      }

      // ⚠️ لا تُنشئ كائن جديد، أعد نفس المرجع
      return prev;
    });
  };

  socket.on("station_update", handleStationChange);
  return () => {
    socket.off("station_update", handleStationChange);
  };
}, []);
//       setTimeout(() => {
//   graphData.nodes = graphData.nodes.map(node => {
//     let list = ['SENIN', 'PRZ', 'SNR'];
     
//     if (list.includes(node.id)) {
//       node.stability = 'stable';
//       node.risk = 'high';
//       console.log("chang: ", node.id , node.stability, node.risk);
//     }
    
//     return node;
//   });
// }, 3000);

  return (
    <div >
      <ToastContainer />
      <ForceGraph2D
        graphData={graphData}
        nodeLabel="id"
        onNodeClick={(node) =>selectStation(node.id)}
        onBackgroundClick={() => selectStation(null)}
    nodeCanvasObject={(node, ctx, globalScale) => {
  const radius = 6;
  const color = getRiskColor(node.risk, node.stability)  ;

  // Draw blue outline if selected
  if (node.id === selectedStation) {
    const outlineRadius = radius + 4;
    ctx.beginPath();
    ctx.arc(node.x, node.y, outlineRadius, 0, 2 * Math.PI, false);
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw node fill
  ctx.beginPath();
  ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
  ctx.fillStyle = color;
  ctx.fill();

  // Node border (optional thin black stroke)
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // Draw label
  const fontSize = 10 / globalScale;
  ctx.font = `${fontSize}px Sans-Serif`;
  ctx.fillStyle = 'black';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(node.id, node.x, node.y + radius + 2);
}}
      />
    </div>
  );
};

export default Graph;