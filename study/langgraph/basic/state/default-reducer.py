from typing import List, Any


class State(dict):
    foo:int
    bar: List[str]
def update_state(current_state:State,updates:dict[str,Any]) -> State:
    new_state = current_state.copy()
    new_state.update(updates)
    return new_state

state:State = {"foo": 1, "bar": ["hi"]}

node1_update = {"foo": 2}

state = update_state(state, node1_update)
print(state) # 输出：{'foo': 2, 'bar': 2}

node2_update = {"bar": ["bye"]}
state = update_state(state,node2_update)
print(state) # 输出：{'foo': 2, 'bar': ['bye']}