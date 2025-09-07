from langgraph.graph import StateGraph, END
from .state import ConversationState
from . import nodes


def build_graph():
    sg = StateGraph(ConversationState)

    # Define nodes
    sg.add_node('intent', nodes.intent_detection_node)
    sg.add_node('slot', nodes.slot_filling_node)
    sg.add_node('validate', nodes.validation_node)
    sg.add_node('execute', nodes.execution_node)

    # Entry -> intent
    sg.set_entry_point('intent')

    # Transitions
    def to_slot(state: ConversationState):
        return 'slot'

    def to_validate(state: ConversationState):
        return 'validate'

    def to_execute(state: ConversationState):
        return 'execute'

    def to_end(state: ConversationState):
        return END

    # Flow: intent -> slot (collect until none missing) -> validate -> execute -> end
    sg.add_edge('intent', 'slot')
    sg.add_conditional_edges('slot', lambda s: 'slot' if s.missing and not s.error else 'validate', {'slot': 'slot', 'validate': 'validate'})
    sg.add_edge('validate', 'execute')
    sg.add_edge('execute', END)

    return sg.compile()
