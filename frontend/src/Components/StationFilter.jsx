 

const SENSORS = ['HE', 'HN', 'HZ'];
const SIGNALS = ['acceleration', 'velocity', 'displacement'];

function StationFilter({ selectedOption, onOptionChange ,selectedSensors = ['HE', 'HN', 'HZ'],setSelectedSensors  }) {
 
  const handleSensorToggle = (sensor) => {
    if (selectedSensors.includes(sensor)) {
      setSelectedSensors(selectedSensors.filter(s => s !== sensor));
    } else {
      setSelectedSensors([...selectedSensors, sensor]);
    }
  };
  
  return (
    <div>
      <ul className="list-group">
         <li className="list-group-item"> Sensors</li>
           <li className="list-group-item"> 
            {SENSORS.map(option=>(
               <>
                <input
                type="checkbox"
                name="option"
                value={option} 
                checked={selectedSensors.includes(option)}
                onChange={()=>handleSensorToggle(option)}
              />
              <label htmlFor={option}  className="ms-1 me-4 text-capitalize">{option}</label> 
              </>
            ))}
           </li>
         <li className="list-group-item"> Mesurment</li>
       
        <li className="list-group-item  ">
          {SIGNALS.map((option) => (
            
          <>
          <span key={option} className="me-3  ">
              <input
                type="radio"
                name="option"
                value={option}
               id={option}
                checked={selectedOption === option}
                onChange={(e) => onOptionChange(e.target.value)}
              />
              <label htmlFor={option}  className="ms-1 text-capitalize">{option}</label>
            </span>
            <br /> 
            
          </>
          ))}
        </li>
      </ul>
    </div>
  );
}

export default StationFilter;
