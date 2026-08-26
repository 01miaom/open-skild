"""List MCAP topics; extraction remains schema-specific."""
import argparse
def main():
 p=argparse.ArgumentParser(); p.add_argument('path'); p.add_argument('--list-topics',action='store_true'); a=p.parse_args()
 try: from mcap.reader import make_reader
 except ImportError: raise SystemExit('Install mcap first: pip install mcap')
 with open(a.path,'rb') as f:
  for channel,schema,_ in make_reader(f).iter_messages(): print(channel.topic,schema.name if schema else '')
if __name__=='__main__': main()
