import numpy as np

class wind_glide_q:
    def __init__(self):                
        # Grid world 크기
        self.num_rows = 7 #행 
        self.num_columns = 10 #열 

        # 바람 정보 (columns의 인덱스에 따른 바람 세기)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        # 상태(state)와 행동(action)의 개수
        self.num_states = self.num_rows * self.num_columns
        self.num_actions = 4

        # Q 테이블 초기화
        self.Q = np.zeros((self.num_states, self.num_actions))

        # 학습 파라미터
        self.learning_rate = 0.5
        self.discount_factor = 0.5
        self.epsilon = 0.1

        
    # 상태 인덱스를 좌표로 변환하는 함수
    def state_to_coordinates(self, state):
        row = state // self.num_columns
        column = state % self.num_columns
        return row, column

    # 좌표를 상태 인덱스로 변환하는 함수
    def coordinates_to_state(self, row, column):
        return row * self.num_columns + column

    def simulation_start(self, num_episodes):
        time_steps =[] 
        time_step = 0
    # Q-Learning 알고리즘
        for episode in range(num_episodes):
            # S(시작) 상태 설정
            
            state = self.coordinates_to_state(3, 0)#4row1col

            # 목표 상태 G의 인덱스
            goal_state = self.coordinates_to_state(3, 7)#4row8col
            
            # 에피소드 진행
            while state != goal_state:
                # e-greedy 정책에 따라 행동 선택
                if np.random.uniform(0, 1) < self.epsilon:
                    action = np.random.randint(self.num_actions)
                else:
                    action = np.argmax(self.Q[state])

                # 선택한 행동 수행
                row, column = self.state_to_coordinates(state)
                next_row, next_column = row, column  # 다음 상태의 좌표 초기화

                if action == 0:  # up
                    next_row -= 1
                elif action == 1:  # down
                    next_row += 1
                elif action == 2:  # right
                    next_column += 1
                elif action == 3:  # left
                    next_column -= 1

                # 바람에 의한 이동(상하)
                next_row += self.wind[column]

                # 경계 체크
                next_row = max(0, min(next_row, self.num_rows - 1))
                next_column = max(0, min(next_column, self.num_columns - 1))

                # 좌표를 상태 인덱스로 변환
                next_state = self.coordinates_to_state(next_row, next_column)

                # 보상과 다음 상태의 정보
                reward = -1

                # 목표 상태에 도달하면 보상을 0으로 설정
                if next_state == goal_state:
                    reward = 0

                # Q 값 업데이트
                self.Q[state, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state, action])

                # 상태 업데이트
                state = next_state
                time_step += 1
            time_steps.append(time_step)
        return time_steps
            

# 학습된 Q 테이블 출력
# print("Learned Q-table:")
# wind = wind_glide_q()
# wind.simulation_start(1000)
# print(wind.Q)






