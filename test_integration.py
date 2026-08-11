import asyncio
import sys
sys.path.insert(0, '.')

# Simulate exactly what smart_brain does
from smart_brain import _handle_device_control

async def test():
    # First call - should work
    result1 = await _handle_device_control('客厅有点热', 'web_session')
    print('Test 1 - 客厅有点热:', result1.get('reply'))
    
    # Reset state
    from device_controller import DEVICE_STATES, state_lock
    async with state_lock:
        for k in DEVICE_STATES:
            if k.endswith('_ac'):
                DEVICE_STATES[k]['power'] = 'off'
    
    # Second call - another text
    result2 = await _handle_device_control('打开卧室空调', 'web_session')
    print('Test 2 - 打开卧室空调:', result2.get('reply'))

asyncio.run(test())
