from mc_gpt_all import MCTall 
import argparse
import os
import yaml
from pprint import pprint
from easydict import EasyDict



def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of Music2Dance')
    parser.add_argument('--config', default='')
    # exclusive arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')
    group.add_argument('--visgt', action='store_true')
    group.add_argument('--anl', action='store_true')
    group.add_argument('--sample', action='store_true')

    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    # ── Inject text_mask_mode into GPT sub-configs (base & head both host attention blocks) ──
    # text_mask_mode is read from the YAML config (field: text_mask_mode, default: full)
    _ALIAS_MAP = {
        'full': 'full', 'time_only': 'time_only', 'part_only': 'part_only', 'none': 'none',
        'no_body_mask': 'time_only', 'no_temporal_mask': 'part_only', 'no_mask': 'none',
    }
    _raw_mode = getattr(config, 'text_mask_mode', 'full')
    _canonical_mode = _ALIAS_MAP.get(_raw_mode, 'full')
    if hasattr(config, 'structure_generate'):
        if hasattr(config.structure_generate, 'base'):
            config.structure_generate.base.text_mask_mode = _canonical_mode
        if hasattr(config.structure_generate, 'head'):
            config.structure_generate.head.text_mask_mode = _canonical_mode
    print(f'[text_mask_mode] raw={_raw_mode!r}  canonical={_canonical_mode!r}')

    agent = MCTall(config)
    print(config)

    if args.train:
        agent.train()
    elif args.eval:
        agent.eval()
    elif args.visgt:
        # print('Wula!')
        agent.visgt()
    elif args.anl:
        # print('Wula!')
        agent.analyze_code()
    elif args.sample:
        config.update({'need_not_train_data':True})
        config.update({'need_not_test_data':True})
        agent.sample()


if __name__ == '__main__':
    main()
