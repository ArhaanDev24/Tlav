import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Tests never touch the real ~/.tlav (state, discovered models, saved chats).
os.environ["TLAV_HOME"] = tempfile.mkdtemp(prefix="tlav-test-")
os.environ["TLAV_NO_UPDATE_CHECK"] = "1"   # and never ask the internet for a newer version
