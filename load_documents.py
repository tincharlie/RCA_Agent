from langchain_core.documents import Document
from services.vector_service import vector_db

docs = [
    Document(
        page_content="""
        High vibration and high bearing temperature indicate bearing failure.
        """
    ),
    Document(
        page_content="""
        Low discharge pressure can be caused by worn bearings.
        """
    ),
    Document(
        page_content="""
        Excessive noise and shaft misalignment can lead to premature bearing wear.
        """
    ),
    Document(
        page_content="""
        High motor current may indicate pump overloading or internal mechanical damage.
        """
    ),
    Document(
        page_content="""
        Low suction pressure may be caused by clogged filters or air leakage in the suction line.
        """
    ),
    Document(
        page_content="""
        Frequent seal failures can result from improper installation or shaft misalignment.
        """
    ),
    Document(
        page_content="""
        Increased vibration at specific frequencies often indicates imbalance in rotating parts.
        """
    ),
    Document(
        page_content="""
        Overheating of motor windings can be caused by poor ventilation or electrical faults.
        """
    ),
    Document(
        page_content="""
        Reduced flow rate may result from impeller wear or blockage in the system.
        """
    ),
    Document(
        page_content="""
        Oil contamination in bearings can lead to increased friction and eventual failure.
        """
    ),
    Document(
        page_content="""
        Cavitation noise and vibration occur due to low suction head or vapor formation in the fluid.
        """
    ),
    Document(
        page_content="""
        Loose mounting bolts can cause excessive vibration and misalignment issues.
        """
    )
]


vector_db.add_documents(docs)