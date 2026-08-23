from models.state import RCAState


def alarm_agent(
        state: RCAState
):
    alarms = [
        "HIGH_VIBRATION",
        "HIGH_BEARING_TEMP",
        "LOW_DISCHARGE_PRESSURE"
    ]

    state["alarm_data"] =(alarms)

    return state