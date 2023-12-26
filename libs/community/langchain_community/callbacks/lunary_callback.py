try:
  import lunary

  LunaryCallbackHandler = lunary.LunaryCallbackHandler
  identify = lunary.identify
except ImportError:
  LunaryCallbackHandler = None
  identify = None
  print("[Lunary] Please install Lunary with `pip install lunary` to use LunaryCallbackHandler") 