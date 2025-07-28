# Per-Server Rate Computation Enhancement

This update adds new functionality to compute reneging and jockeying rates per server and per information source, enabling more granular analysis of queue behavior.

## New Methods

### `compute_reneging_rate_per_server(subscribers, server_id)`

Computes the reneging rate for requests added to a specific server using the correct subscriber list.

**Parameters:**
- `subscribers`: List of request objects (either `state_subscribers` or `nn_subscribers`)
- `server_id`: Server identifier ("1" or "2")

**Returns:** Float between 0 and 1 representing the reneging rate

### `compute_jockeying_rate_per_server(subscribers, server_id)`

Computes the jockeying rate for requests added to a specific server using the correct subscriber list.

**Parameters:**
- `subscribers`: List of request objects (either `state_subscribers` or `nn_subscribers`)
- `server_id`: Server identifier ("1" or "2")

**Returns:** Float between 0 and 1 representing the jockeying rate

### `get_rates_summary_per_server_and_source()`

Returns a comprehensive summary of all rates organized by server and information source.

**Returns:** Dictionary with structure:
```python
{
    "server_1": {
        "state_subscribers": {
            "reneging_rate": float,
            "jockeying_rate": float,
            "count": int
        },
        "nn_subscribers": {
            "reneging_rate": float,
            "jockeying_rate": float,
            "count": int
        }
    },
    "server_2": { ... }
}
```

## Enhanced Visualization

### `plot_rates_per_server_and_source()`

Creates enhanced visualizations showing rates broken down by both server and information source, with bar charts comparing state subscribers vs NN subscribers.

## Request Object Enhancement

All `Request` objects now include a `server_id` property that indicates which server ("1" or "2") the request was assigned to.

## Example Usage

```python
from activity import RequestQueue

# Create queue and add requests
requestObj = RequestQueue(utility_basic=1.0, discount_coef=0.1)
for i in range(10):
    requestObj.addNewRequest(expected_time_to_service_end=8.0, batchid=i)

# Compute specific rates
server_1_state_reneging = requestObj.compute_reneging_rate_per_server(
    requestObj.state_subscribers, "1"
)
server_2_nn_jockeying = requestObj.compute_jockeying_rate_per_server(
    requestObj.nn_subscribers, "2"
)

# Get comprehensive summary
summary = requestObj.get_rates_summary_per_server_and_source()
print(summary)

# Generate enhanced visualizations
requestObj.plot_rates_per_server_and_source()
```

## Information Sources

The system distinguishes between two types of information sources:

- **State Subscribers**: Requests that use raw queue state information
- **NN Subscribers**: Requests that use neural network-based knowledge

This distinction allows for targeted analysis of how different information sources affect customer behavior patterns.

## Backward Compatibility

All existing methods remain functional. The new methods provide additional granularity while maintaining compatibility with existing code.

## Files Modified

- `src/activity.py`: Added new methods and enhanced plotting
- `src/ver_ren_jock.py`: Added new methods and updated usage examples
- `tests/unittests.py`: Added comprehensive test coverage
- Examples: `comprehensive_example.py` and `test_per_server_rates.py`

## Benefits

1. **Granular Analysis**: Separate analysis by server and information source
2. **Targeted Optimization**: Optimize strategies per information source
3. **Enhanced Visualization**: Better plots showing detailed breakdowns
4. **Research Insights**: Support for studying impact of different information types
5. **Backward Compatibility**: Existing code continues to work unchanged