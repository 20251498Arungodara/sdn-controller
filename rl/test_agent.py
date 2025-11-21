from rl.agent import CAMBEAgent

def main():
    agent = CAMBEAgent(alpha=0.5, gamma=0.9, eps=0.5, tau=0.8, seed=42)
    state = (0,0,0)
    Avalid = [0,1,2]
    # pick actions a few times
    print("Selecting actions (5 samples):")
    for _ in range(5):
        a = agent.select_action(state, Avalid)
        print(" selected:", a)
    # update Q for one transition
    next_state = (1,0,0)
    agent.update(state, 0, reward=-10.0, next_state=next_state, next_actions=Avalid)
    print("After update, Q[(state,0)] = ", agent.get_q(state, 0))
    # save/load test
    agent.save('rl/test_agent_q.pkl')
    agent2 = CAMBEAgent()
    agent2.load('rl/test_agent_q.pkl')
    print("Loaded agent Q size:", len(agent2))

if __name__ == '__main__':
    main()
