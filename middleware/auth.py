from fastapi import Header
from fastapi import HTTPException

def validate_token(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Token")
    return True


