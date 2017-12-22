# myTensorUtil
这是一个测试用 TensorUtil 仓库

## 使用方法
1. 下载本仓库到本地

2. 在 `tensorUtil.cu` 和 `tensorUtil.h` 中编写你的代码

3. 在命令行中输入
   ```
   make
   ```
   编译程序。如果想加入 debug 符号表，使用
   ```
   make DEBUG=1
   ```
   清除项目目标文件以便重新编译，使用
   ```
   make clean
   ```

4. Debug 时，需要先使用 `make DEBUG=1` 生成项目，然后使用 `gdb ./testtu` 打开调试器，下面是一些常用命令：
   ```
   r : 开始运行/重新运行;
   b <line-number> :  在行号 <line-number> 处打断点;
   d <breakpoint-number> : 删除断点编号为 <breakpoint-number> 的断点;
   c : 从断点处接着运行;
   n : 按行执行;
   s : 按步执行（遇到函数会进入）;
   bt: 打印调用栈;
   q : 退出
   ```

   如果遇到了 cudaError 的错误，可使用 `cuda-memcheck ./testtu` 对程序对显存的访问进行检查，检查时图形界面有可能会卡住。

5. 使用 `./testtu` 执行代码

## 任务清单
1. 张量切片操作

   在 `tensorUtil.cu` 中完成张量切片操作 `sliceTensor` 。操作所需的张量定义、辅助函数均已写好，只需完成 `sliceTensor` 即可，在 `sliceTensor` 函数中调用 CUDA 核函数 `sliceTensorkernel` 。代码写完后，编译并执行 `./testtu` 会执行一个测试用例，在屏幕上显示执行结果。