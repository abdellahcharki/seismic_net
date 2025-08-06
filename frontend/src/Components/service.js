import axios from "axios";

function url(path) {
  return "http://localhost:5000/" + path;
}

export function getAllStations() {
  return axios.get(url("hybrid/nodes"));
}

export function getAllLinks() {
  return axios.get(url("hybrid/links"));
}

export function getStationById(id) {
  return axios.get(url(`hybrid/station/${id}`));
}

export function getRiskColor(risk, stab) {
  if (stab === "unstable") return "red";
  else if (stab === "stable" && risk == "high") return "orange";
  else if (stab === "stable" && risk == "low") return "#3CB043";
  else return "gray";
}
