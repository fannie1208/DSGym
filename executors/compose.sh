# python generate_compose.py -n 2 \
#     --types  "executor-prebuilt:2" \
#     -m /Users/junlinwang/Documents/Research/Agents/SkyRL/skyrl-gym/skyrl_gym/envs/allocated_code/discovery_bench/data \
#     -o docker-discoverybench.yml

# sudo docker compose -f docker-discoverybench.yml up -d --build