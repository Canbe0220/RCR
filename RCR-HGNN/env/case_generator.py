import random
import time
import numpy as np

class CaseGenerator:
    '''
    FJSP instance generator - Integrated with SD2 Logic
    '''
    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path='data/',
                 flag_same_opes=True, flag_doc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path
        self.job_init = job_init
        self.num_mas = num_mas

        self.mas_per_ope_min = 1
        self.mas_per_ope_max = num_mas
        self.opes_per_job_min = opes_per_job_min
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1
        self.proctime_per_ope_max = 99

    def get_case(self, idx=0):
        """
        Fully equivalent to SD2_instance_generator + matrix_to_text,
        while keeping the original second-stage return format.
        """

        # ===== 1. 参数对齐 =====
        n_j = self.job_init
        n_m = self.num_mas

        op_per_job = n_m
        
        low = self.proctime_per_ope_min
        high = self.proctime_per_ope_max

        data_suffix = getattr(self, "data_suffix", "mix")

        # ===== 2. 可选机器范围 =====
        op_per_mch_min = 1
        if data_suffix == "nf":
            op_per_mch_max = 1
        elif data_suffix == "mix":
            op_per_mch_max = n_m
        else:
            op_per_mch_min = getattr(self, "op_per_mch_min", 1)
            op_per_mch_max = getattr(self, "op_per_mch_max", n_m)

        if op_per_mch_min < 1 or op_per_mch_max > n_m:
            raise ValueError(f'[{op_per_mch_min},{op_per_mch_max}] invalid for num_mch={n_m}')

        # ===== 3. 工序数 =====
        n_op = int(n_j * op_per_job)
        job_length = np.full(shape=(n_j,), fill_value=op_per_job, dtype=int)

        # ===== 关键：随机顺序与第一段一致 =====
        op_use_mch = np.random.randint(
            low=op_per_mch_min,
            high=op_per_mch_max + 1,
            size=n_op
        )

        op_pt = np.random.randint(
            low=low,
            high=high + 1,
            size=(n_op, n_m)
        )

        for row in range(op_pt.shape[0]):
            mch_num = int(op_use_mch[row])
            if mch_num < n_m:
                inf_pos = np.random.choice(
                    np.arange(0, n_m),
                    n_m - mch_num,
                    replace=False
                )
                op_pt[row][inf_pos] = 0

        op_per_mch = np.mean(op_use_mch)

        # ===== 4. 生成文本格式 =====
        lines = []
        lines_doc = []

        line0 = '{0}\t{1}\t{2}\n'.format(n_j, n_m, op_per_mch)
        lines.append(line0)
        lines_doc.append(line0.strip())

        op_idx = 0
        for j in range(n_j):
            line = f'{job_length[j]}'
            for _ in range(job_length[j]):
                use_mch = np.where(op_pt[op_idx] != 0)[0]
                line += ' ' + str(use_mch.shape[0])
                for k in use_mch:
                    line += ' ' + str(k + 1) + ' ' + str(op_pt[op_idx][k])
                op_idx += 1
            lines.append(line + '\n')
            lines_doc.append(line)

        # ===== 5. 保存文件 =====
        if self.flag_doc:
            import os
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            file_path = self.path + '{0}j_{1}m_{2}.fjs'.format(
                n_j, n_m, str(idx + 1).zfill(3)
            )
            with open(file_path, 'a') as doc:
                for l in lines_doc:
                    doc.write(l + '\n')

        # ===== 6. 返回值保持和原第二段一致 =====
        return lines, self.job_init, self.job_init

class CaseGenerator_SD1:
    '''
    FJSP instance generator
    '''
    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path='../data/',
                 flag_same_opes=True, flag_doc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = num_mas
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.proctime_per_ope_max = 20
        self.proctime_dev = 0.2

    def get_case(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        '''
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]
        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas+1), val))
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)]
        for i in range(len(self.nums_option)):
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i]+self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')
        if self.flag_doc:
            doc = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx+1),3)),'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_jobs
