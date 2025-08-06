
 
import logo from "../logo.png"
function NavScrollExample({ show, setShow }) {
  return (
    <nav className="navbar navbar-expand-lg  ">
      <div className="container-fluid">
        <a className="navbar-brand" href="#"><img height={45} src={logo} alt="" srcset="" /> Seismic Net </a>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarScroll" aria-controls="navbarScroll" aria-expanded="false" aria-label="Toggle navigation">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarScroll">
          <ul className="navbar-nav me-auto my-2 my-lg-0"    >
          </ul>
          <form className="d-flex" role="search">
            <button className="btn btn-outline-light" onClick={() => setShow(!show)} type="button"> {!show ? <><i class="fas fa-project-diagram"></i> Graph</> : <><i class="fas fa-map    "></i> Map</>}</button>
          </form>
        </div>
      </div>
    </nav>

  );
}

export default NavScrollExample;