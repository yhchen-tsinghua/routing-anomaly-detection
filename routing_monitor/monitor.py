import pandas as pd
import ipaddress

class Monitor:
    class Node:
        def __init__(self):
            self.routes     = dict() # forwarder -> aspath
            self.left       = None
            self.right      = None

        def get_left(self):
            if self.left is None:
                self.left = Monitor.Node()
            return self.left

        def get_right(self):
            if self.right is None:
                self.right = Monitor.Node()
            return self.right

        def find_route(self, forwarder):
            if forwarder in self.routes:
                return self.routes[forwarder]
            return None

    def __init__(self):
        self.root = Monitor.Node()
        self.route_changes = []

    def update(self, timestamp, prefix_str, vantage_point, aspath_str, detect):
        prefix = ipaddress.ip_network(prefix_str)

        if prefix.version == 6: return
        prefixlen = prefix.prefixlen
        prefix = int(prefix[0]) >> (32-prefixlen)

        aspath = aspath_str.split(" ")
        forwarder = aspath[0] # NOTE: forwarder could be vantage point, or could not

        n = self.root
        original_route = None
        for shift in range(prefixlen-1, -1, -1): # find the original route
            left = (prefix >> shift) & 1

            if left: n = n.get_left()
            else: n = n.get_right()
            
            if n.find_route(forwarder) is not None:
                original_route = [shift, n.find_route(forwarder)]

        if detect and original_route is not None:
            shift, original_path = original_route
            vict_prefix = ipaddress.ip_network(prefix_str) \
                            .supernet(new_prefix=prefixlen-shift)
            if aspath != original_path:
                self.route_changes.append({
                    "timestamp"    : timestamp,
                    "vantage_point": vantage_point,
                    "forwarder"    : forwarder,
                    "prefix1"      : str(vict_prefix),
                    "prefix2"      : prefix_str,
                    "path1"        : " ".join(original_path),
                    "path2"        : " ".join(aspath),
                })

        n.routes[forwarder] = aspath

    def consume(self, df, detect=False):
        if "A/W" in df.columns:
            df = df.loc[df["A/W"] == "A"] # NOTE: fair move
        cols = ["timestamp", "prefix", "peer-asn", "as-path"]

        for a in df[cols].values:
            self.update(*a, detect=detect)
