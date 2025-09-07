import React from 'react'
import { ButtonGroup, Button } from 'reactstrap'

export default function ModeToggle({ mode, onChange }){
  return (
    <ButtonGroup>
      <Button color={mode === 'onboarding' ? 'light' : 'secondary'} onClick={() => onChange('onboarding')}>Onboarding</Button>
      <Button color={mode === 'agent' ? 'light' : 'secondary'} onClick={() => onChange('agent')}>Agent Chat</Button>
    </ButtonGroup>
  )
}
