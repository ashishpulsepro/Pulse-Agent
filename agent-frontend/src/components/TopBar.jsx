import React from 'react'
import { Navbar, NavbarBrand } from 'reactstrap'

export default function TopBar(){
  return (
    <Navbar className="bw-navbar" dark expand="md">
      <NavbarBrand href="#" className="fw-semibold">Pulse Agent</NavbarBrand>
    </Navbar>
  )
}
