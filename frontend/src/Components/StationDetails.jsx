
 
function StationDetails({station}) {
  return (
 
    <div>
              <ul className="list-group">
                <li className="list-group-item"><strong>ID:</strong> {station.id}</li>
                <li className="list-group-item"><strong>Latitude:</strong> {station.latitude}</li>
                <li className="list-group-item"><strong>Longitude:</strong> {station.longitude}</li>
                <li className="list-group-item"><strong>Risk:</strong> {station.risk}</li>
                <li className="list-group-item"><strong>Risk Score:</strong> {station.risk_score}</li>
                <li className="list-group-item"><strong>Stability:</strong> {station.stability}</li>
                <li className="list-group-item"><strong>Stability Score:</strong> {station.stability_score}</li>
              </ul>
            </div>
  );
}

export default StationDetails;