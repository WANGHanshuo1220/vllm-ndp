from typing import List, Dict, Tuple, Set, TypeAlias
import asyncio

class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._new_requests = asyncio.Queue()
        self.new_requests_event = asyncio.Event()
        self.finished_request = asyncio.Queue()
        self.all_recieved_requests = []

    # def __contains__(self, item):
    #     return item in self._request_streams

    # def __len__(self) -> int:
    #     return len(self._request_streams)

    def add_request(self, seq_id: int, content) -> None:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        # if seq_id in self.all_recieved_requests:
        #     raise KeyError(f"Sequence {seq_id} already exists.")

        self.all_recieved_requests.append(seq_id)

        self._new_requests.put_nowait(content)

        self.new_requests_event.set()

    # def get_new_and_aborted_requests(self) -> Tuple[List[Dict], Set[str]]:
    #     """Get the new requests and finished requests to be
    #     sent to the engine."""
    #     new_requests: List[Dict] = []
    #     finished_requests: Set[str] = set()

    #     while not self._aborted_requests.empty():
    #         request_id = self._aborted_requests.get_nowait()
    #         finished_requests.add(request_id)

    #     while not self._new_requests.empty():
    #         stream, new_request = self._new_requests.get_nowait()
    #         request_id = stream.request_id
    #         if request_id in finished_requests:
    #             # The request has already been aborted.
    #             stream.finish(asyncio.CancelledError)
    #             finished_requests.discard(request_id)
    #         else:
    #             self._request_streams[request_id] = stream
    #             new_requests.append(new_request)

    #     return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()