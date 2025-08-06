
function InfoSide() {
    return (
        <div className="infoside">
            <h1>Fraud Detection</h1>
            <hr />
            <div>
                <input type="radio" className="btn-check" name="options" id="option1" autocomplete="off"   />
                <label className="btn  " for="option1">  Map View </label>

                <input type="radio" className="btn-check" name="options" id="option2" autocomplete="off" />
                <label className="btn  " for="option2"> View Graph</label>
 
            </div>
            <h5 class="card-title mb-0">Person Information</h5>
            <hr />
            <p><strong>Name:</strong> CAMILLA V. CARR</p>
            <p><strong>Role(s):</strong> Driver, Witness, Passenger, Passenger</p>
            <p><strong>Enter Date:</strong> Thu Aug 19 2021 02:00:00 GMT+0200</p>
            <p><strong>Exit Date:</strong> Sat Apr 15 2023 02:00:00 GMT+0200</p>
        </div>
    )

}



export default InfoSide;