import numpy as np
import matplotlib.pyplot as plt
from wind_glide_q import wind_glide_q
from wind_glide_sarsa import wind_glide_sarsa



# Main 함수
if __name__ == "__main__":
    num_episodes = 170  # 에피소드 수 설정
    
    # Sarsa 학습  # Q-learning 학습
    Sarsa = wind_glide_sarsa()
    q = wind_glide_q()
    time_steps_sarsa = Sarsa.simulation_start(num_episodes)
    Sarsa.print_optimal_policy()

    time_steps_q_learning = q.simulation_start(num_episodes)
    q.print_optimal_policy()

    # 에피소드별 시간 단계 수 그래프 그리기
    episodes = np.arange(1, num_episodes + 1)
    plt.plot(time_steps_sarsa, episodes,label='Sarsa')
    plt.plot(time_steps_q_learning, episodes, label='Q-learning')
    plt.xlabel('Time Steps')
    plt.ylabel('Episodes')
    plt.legend()
    plt.savefig('savefig_default.png')
    