//1、 arr = [  从左到右，递增， 从上到下，递增
//     [1, 2, 8, 9],
//     [2, 4, 9, 12],
//     [4, 7, 10, 13],
//     [6, 8, 11, 15]
// ]
// 找出值
function findNum(matrix, num, rows, columns) {
    let found = false
    if (matrix !== null && rows > 0 && columns > 0) {
        let row = 0
        let column = columns - 1
        while(row < rows && column >=0) {
            if(matrix[row][column] > num) {
                column--
            } else if (matrix[row][column] < num) {
                row++
            } else {
                found = true
                console.log({row, column})
                break
            }
        }
    }
    return found
}

//2、 替换空格为%20
// str: we are happy
function replaceBlock(str) {
    let len = str.length
    let arr = str.split('')
    // 先扫描一遍多少空格
    let blockCnt = 0
    for(let i = 0; i < arr.length; i++) {
        if(arr[i] === ' ') blockCnt++
    }
    
    // 双指针,一个指向原来字符串的末尾，一个指向替换后字符串的末尾
    let p1 = len - 1
    let p2 = len + blockCnt * 2 - 1
    // 根据结果的长度扩充数组
    arr.length = len + blockCnt * 2
    while(p1 < p2 && p1 >=0) {
        if(arr[p1] !== ' ') {
            arr[p2] = arr[p1]
            p1--
            p2--
        } else if(arr[p1] === ' ') {
            arr[p2] = '0'
            arr[p2 - 1] = '2'
            arr[p2 -2] = '%'
            p2 -= 3
            p1--
        }
    }

    return arr.join('')
}

//3、 链表删除节点
function removeNode(pHead, val) {
    if(pHead === null) {
        return null
    }

    if(pHead.val === val) {
        pHead = pHead.next
    } else {
       let node = head
       while(node.next !== null && node.next.val !== val)  {
        node = node.next
       }

       if(node.next !== null && node.next.val === val) {
        pToDeleted = node.next
        node.next = node.next.next
       }
    }
    return pHead
} 

//4、 从尾到头打印
function printListReverse(pHead) { // 使用栈
    let stack = []
    let node = pHead
    while(node !== null) {
        stack.push(node.val)
        node = node.next
    }

    while(stack.length) {
        console.log(stack.pop())
    }
}

// 递归本来就是栈结构，所以可以递归实现
function printListReverse(pHead) {
    if(pHead !== null) {
        if(pHead.next !== null) {
            printListReverse(pHead.next)
        }
        console.log(pHead.value)
    }
}

//5、 重建二叉树， 给出前序遍历和中序遍历结果
const BinaryTreeNode = (value) => {
    this.value = value;
    this.left = null;
    this.right = null;
}
function constructTree(preOrder, midOrder) {
    if(!preOrder || !midOrder || preOrder.length !== midOrder.length) {
        return null
    }

    const core = (preOrder, midOrder) => {
        // 前序第一个值是根节点
        const rootValue = preOrder[0]
        const root = new BinaryTreeNode(rootValue)

        // 找到中序遍历中根节点的位置
        const rootIndexInMidorder = midOrder.indexOf(rootValue)

        // 中序遍历根节点左边是左子树，右边是右子树
        // 构建左子树
        const leftMidOrder = midOrder.sclie(0, rootIndexInMidorder)
        const leftPreOrder = preOrder.sclie(1, 1 + leftMidOrder.length)
        if(leftMidOrder.length) {
            root.left = core(leftPreOrder, leftMidOrder)
        }

        // 构建右子树
        const rightMidOrder = midOrder.sclie(rootIndexInMidorder + 1)
        const rightPreOrder = preOrder.sclie(1 + leftMidOrder.length)
        if(rightMidOrder.length) {
            root.right = core(rightPreOrder, rightMidOrder)
        }
    }

    return core(preOrder, midOrder)
}

// 栈实现， 不太懂需研究
function constructTree(preOrder, midOrder) {
    if(!preOrder || !midOrder || preOrder.length !== midOrder.length) {
        return null
    }

    let preOrderIndex = 1
    let midOrderIndex = 0
    let root =  new BinaryTreeNode(preOrder[preOrderIndex])
    let stack = []
    let cur = root

    while(preOrderIndex < preOrder.length) {
        // 前序的根，不等于中序的当前的值，说明中序当前的值一定在根的左子树里
        if(cur.value !== midOrder[midOrderIndex]) {
            // 前序遍历可以确定确定左节点，所以可以确定当前前序所指的节点是当前根的左节点
            cur.left = new BinaryTreeNode(preOrder[i])
            stack.push(cur)
            cur = cur.left
        } else { // 在右子树里
            midOrderIndex++

            // 栈顶等于当前中序遍历值
            while(stack.length > 0 && stack[stack.length - 1].value === midOrder[midOrderIndex]) {
                cur = stack.pop()
                midOrderIndex++
            }

            cur.right = new BinaryTreeNode(preOrder[i])
            cur = cur.right
        }

        preOrderIndex++
    }

    return root
}


//6、 用两个栈实现队列
class myQuene {
    constructor() {
        this.stack1 = [] // 存入队数据
        this.stack2 = [] //存出队顺序的数据
    }

    addTail(ele) {
        this.stack1.push(ele)
    }

    deleteHead() {
        if(this.stack2.length === 0) {
            while(this.stack1.length) {
                this.stack2.push(this.stack1.pop())
            }
        }

        if(this.stack2.length === 0) {
            throw new Error('empty')
        }

        return this.stack2.pop()
    }
} 

//7、 查找数组中重复的数字

// 排序
function findRepeatNum(numbers) {
    numbers.sort((a, b) => a-b)
    for(let i = 1; i < numbers.length; i++ ) {
        if(numbers[i] === numbers[i - 1]) {
            return numbers[i]
        }
    }
    return null
}

// hash
function findRepeatNum(numbers) {
    const map = new Map()
    for(let number of numbers) {
        if(map.has(number)) {
            return number
        } else {
            map.set(number, 1)
        }
    }
    return null
}

// 原地交换法, 核心思想是把每一项放到值跟数组下标相等的位置，如果交换时与目标位置相等，则代表已经找到了
function findRepeatNum(numbers) {
    for(let i = 0; i < numbers.length; i++) {
        while(numbers[i] !== i) { // 确保当前处理的元素还没有到达其对应的位置， 只要numbers[i]不等于i，就继续循环
            // 检查当前元素是否与它应该在的位置上的元素相等。如果相等，说明找到了重复的元素，直接返回该元素
            if(numbers[i] === numbers[numbers[i]]) {
                return numbers[i]
            }

            [numbers[i], numbers[numbers[i]]] = [numbers[numbers[i]], numbers[i]]
        }
    }
    return null
}

// 也可以用快慢指针实现，把数组当成一个环形链表，最终在数字相同时停下来，不过好像没有环的时候就停不下来了。。。
function findRepeatNum(numbers) {
    // 初始化快慢指针，都从第一个元素开始
    let slow = numbers[0];
    let fast = numbers[0];
    // 移动快慢指针，快指针速度是慢指针的两倍
    do {
        slow = numbers[slow];
        fast = numbers[numbers[fast]];
    } while (slow!== fast);
    // 找到环之后，将慢指针重新设置为起始位置
    slow = numbers[0];
    // 快慢指针以相同速度移动，找到环的入口
    while (slow!== fast) {
        slow = numbers[slow];
        fast = numbers[fast];
    }
    return slow;
}


//8、找出数组中的重复元素，数组元素范围在 1 到 n 之间，数组长度为 n + 1，不改变原数组
// 分析，数组的值都在1到n之间如果，如果没有重复的值，那么数组长度为n，但是现在时n+1，肯定有重复的值
// 那么我们可以对值进行二分，假设把值区间氛围1-m和m+1-n之间
// 如果遍历数组，找到1到m间的数量大于m，那么肯定在这里，否则在m+1-n之间
// 于是可以通过这样对值的二分逐渐缩小范围，找到具体的重复的值
function findRepeatNum(numbers) {

    if(!numbers) return -1

    // 计算区间内数组里符合值的个数
    const countRange = (numbers, start, end) => {
        if(!numbers) return 0

        let count = 0

        for(const num of numbers) {
            if(num >= start && num <= end) {
                count++
            }
        }

        return count
    }

    let start = 1
    let end = numbers.length - 1 // 数组长度为 n+1，所以值的最大值为数组长度-1

    while(end >= start) {
        let mid = end + start >> 1

        let count = countRange(numbers, start, mid)

        if(end === start) { //缩小到一个元素
            if(count > 1) { // 找到了
                return start
            } else {
                break
            }
        } 

        // count 大于不重复时应该的元素数，代表在这里面
        if(count > (mid - start + 1)) {
            end = mid
        } else {
            start = mid + 1
        }
    }

    return -1
}


// 9.找到树的下一个节点：给一个二叉树中的节点，求出中序遍历中的下一个节点，不要整个遍历一遍
// 中序遍历有几个规律
// a、如果当前节点有右子树， 那么它的下一个节点是沿着右子树的left节点往下找最后的那个
// b、如果当前节点没有右子树，如果它是它的父节点的左子节点，那么它的下一个节点就是它的父节点
// c、如果当前节点没有右子树， 如果它是它的父节点的右子节点，那么往上层找，找到一个节点满足该节点是自己父节点的左子节点，那么下一个节点就是找到的这个节点的父节点
//  BinaryTreeNode =  {
//     value,
//     left,
//     right,
//     parent
// }
function findNextNode(node) {
    if(node === null) {
        return null
    }

    if(node.right) {
        let right = node.right
        while(right.left) {
            right = right.left
        }

        return right
    } else {
        let parent = node.parent

        if(parent === null) {
            return null
        }

        if(parent.left === node) {
            return parent
        } else {
            while(parent.parent.left !== parent && parent.parent !== null) {
                parent = parent.parent
            }

            return parent.parent
        }
    }
}


//10、斐波那契数列
// 递归，不推荐
function fibonacci(n) {
    if(n < 2) return n
    return fibonacci(n - 1) + fibonacci(n - 2)
}
// 迭代法
function fibonacci(n) {
    if(n < 2) return n

    let fib1 = 0
    let fib2 = 1
    let fibn
    for(let i = 2; i <= n; i++) {
        fibn = fib1 + fib2
        fib1 = fib2
        fib2 = fibn
    }

    return fibn
}

// 11、青蛙跳台阶, 一只青蛙一次可以跳上 1 级台阶，也可以跳上 2 级台阶。求该青蛙跳上一个n级的台阶总共有多少种跳法
// 可以发现，青蛙跳台阶的问题与斐波那契数列密切相关。当n = 1时，只有一种跳法；当n = 2时，有两种跳法；
// 当n > 2时，青蛙可以从第n - 1级台阶跳一级上来，也可以从第n - 2级台阶跳两级上来，
// 因此跳法总数为 f(n) = f(n-1) + f(n-2)，这正是斐波那契数列的递推公式。
function frogJump(n) {
    if(n <= 2) return n

    let fib1 = 1
    let fib2 = 2
    let fibn
    for(let i = 3; i <= n; i++) {
        fibn = fib1 + fib2
        fib1 = fib2
        fib2 = fibn
    }

    return fibn
}

// 12、找出旋转数组最小的数字
// 给定一个递增排序的数组的一个旋转，输出旋转数组的最小元素
// 例如，数组[3, 4, 5, 1, 2]是[1, 2, 3, 4, 5]的一个旋转，该数组的最小元素为1。
// a、可以考虑到旋转数组的特性，数组中会有两个单调的子数组，其交界处的节点就是最小值，在最小值左边元素单调递增，右边及右边的元素单调递增，左边的数组元素都大于右边的数组元素
// b、 可以使用两个指针，p1指向数组首位，p2指向数组末尾
// c、 算出中间位置mid
//    如果mid位置元素大于等于p1，代表mid位于前面的递增数组里，最小值应该位于mid元素的后面，这个时候把p1移动到mid处，
//    如果mid位置元素小于等于p2，代码mid位于后面的递增数组里，最小值应该位于mid元素的前面。这个时候把p2移动到mid处
// p1始终指向前面的递增数组，p2始终处于后面的递增数组，当p1指向左边的最后一个元素，p2指向右边的第一个元素时，代表找到了最小值，循环结束
// 结束条件也可设为p1 > p2
//例外状态：假如数组为[1, 0, 1, 1, 1]时中间值为1，1 >= 1, p1会移动到2的位置，这显然有问题，所以需要追加判断条件
// 如果p1、p2、mid指向的数字相等，则以p1、p2为范围找出最小值
function findMin(numbers) {
    const minOrder = (numbers, p1, p2) => {
        let res = numbers[p1]
        for(let i = p1; i <= p2; i++) {
            if(numbers[i] < res) {
                res = numbers[i]
            }
        }
        return res
    }

    let len = numbers.length
    if(!numbers || !len) {
        return null
    }

    let p1 = 0, p2 = len - 1
    let mid = 0 // 在数组未旋转的情况下把mid初始化为0，就不必进入循环了
    while(numbers[p1] >= numbers[p2]) {
        // 结束条件， p1指向左边的最后一个元素，p2指向右边的第一个元素时，代表找到了最小值，循环结束
        if(p2 - p1 === 1) { 
            mid = p2
            break
        }

        mid = p1 + p2 >> 1

        // 假如数组为[1, 0, 1, 1, 1]时中间值为1，1 >= 1, p1会移动到2的位置，这显然有问题，所以需要追加判断条件
        // 如果p1、p2、mid指向的数字相等
        if(numbers[p1] === numbers[p1] && numbers[p1] === numbers[mid]) {
            return minOrder(numbers, p1, p2)
        }

        if(numbers[mid] >= numbers[p1]) {
            p1 = mid
        } else if(numbers[mid] <= numbers[p2]) {
            p2 = mid
        }
    }

    return numbers[mid]
}

// 13、矩阵中的路径
// 给出["a","b","c","e"],["s","f","c","s"],["a","d","e","e"]，寻找是否存在"abcced"
function hasPath(matrix, str) {
    const hasPathCore = (matrix, rows, columns, row, column, str, pathLength, visited) => {
        if (pathLength === str.length) {
            return true;
        }
        // 检查格子坐标合法，未被访问过，且值和当前应该该访问的值相等
        if (row >= 0 && row < rows && column >= 0 && column < columns && !visited[row][column] && matrix[row][column] === str[pathLength]) {
            visited[row][column] = true;
            // 往四个方向访问
            const res = hasPathCore(matrix, rows, columns, row - 1, column, str, pathLength + 1, visited) ||
                hasPathCore(matrix, rows, columns, row + 1, column, str, pathLength + 1, visited) ||
                hasPathCore(matrix, rows, columns, row, column + 1, str, pathLength + 1, visited) ||
                hasPathCore(matrix, rows, columns, row, column - 1, str, pathLength + 1, visited);

            if (res) {
                return true;
            }

            // 该条路径不符合回溯
            visited[row][column] = false;
        }
        return false;
    }
    let rows = matrix.length;
    let columns = matrix[0].length;
    if (!matrix || rows < 1 || columns < 1 || !str || !str.length) {
        return false;
    }

    // 用来记录每个节点的访问情况
    const visited = new Array(rows).fill(0).map(() => new Array(columns).fill(false));
    // 以任何一个点为起始位置开始进行匹配
    for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
            if (hasPathCore(matrix, rows, columns, row, column, str, 0, visited)) {
                return true;
            }
        }
    }

    return false;
}



// 14、机器人运动范围
// m * n 的方格，初始坐标为（0，0），但不能进入行坐标列坐标位数之和大于k的格子,请问机器人可以进入多少个格子
// 比如k为18，可以进入（35，37）的格子，但是不能进入（35，38）的格子，因为加起来是19
function moveCount(rows, columns, k) {
    function sumOfDigits(row, column) {
        return [...String(row)].reduce((sum, char) => sum + parseInt(char, 10), 0) +
               [...String(column)].reduce((sum, char) => sum + parseInt(char, 10), 0);
    }

    const countCore = (rows, columns, column, row, k) => {
        if (row >= 0 && row < rows && column >= 0 && column < columns && !visited[row][column] && sumOfDigits(row, column) <= k) {
            visited[row][column] = true;
            const res = countCore(rows, columns, column, row - 1, k) +
                        countCore(rows, columns, column, row + 1, k) +
                        countCore(rows, columns, column + 1, row, k) +
                        countCore(rows, columns, column - 1, row, k);

            return res + 1;
        }

        return 0;
    }

    if (k <= 0 || rows <= 0 || columns <= 0) {
        return 0;
    }

    const visited = new Array(rows).fill(0).map(() => new Array(columns).fill(false));
    return countCore(rows, columns, 0, 0, k);
}

// 15、剪绳子
// 给你一根长度为n的绳子，请把绳子剪成m段（m、n都是整数，n > 1并且m > 1），
// 每段绳子的长度记为k[0]，k[1]，...，k[m]。请问k[0] × k[1] ×... × k[m]可能的最大乘积是多少？
// 例如，当绳子的长度是 8 时，把它剪成长度分别为 2、3、3 的三段，此时得到的最大乘积是 18
// 动态规划的特点：
// 1、求最优解 
// 2、整体的最优解依赖各个子问题的最优解 
// 3、把大问题分解成若干个小问题，这些小问题之间还有相互重叠
// 4、从下往上顺序先计算小问题的最优解并存储下来，在以此为基础求大问题的最优解，从上往下分析问题，从下往上求解问题

// 贪婪算法的特点：
// 每一步都可以做出一个贪婪选择，基于这个选择，我们确定能够得到最优解

// 动态规划

// 定义f(n)为建成若干段长度乘积的最大值
// 如果长度n，可以减成n - 1段，第1次有n - 1种选择，也就是剪出来的第一段绳子可能长1，2，3，...，n - 1
// 因此f(n) = max(f(i) * f(n-i))，0<i<n
// 从下而上求解：
function cutString(n) {
    if(n < 2) {
        return 0
    }

    if(n === 2) {
        return 1
    }

    if(n === 3) {
        return 2
    }

    // 创建数组存储中间结果
    const resArr = new Array(n + 1).fill(0)
    // 这三个初始值是
    resArr[1] = 1
    resArr[2] = 2
    resArr[3] = 3
    for(let i = 4; i <= n; i++) {
        let max = 0
        // 将长度为i的绳子减成两半第一段的长度为j，之所以i / 2，是因为绳子的对称性，剪成1和5和5和1，最终的乘积是相同的
        for(let j = 1; j <= Math.floor(i/2); j++ ) { 
            const res = resArr[j] * resArr[i - j]
            if(max < res) {
                max = res
            }
        }
        resArr[i] = max
    }

    return resArr[n]
}

// 贪婪算法
// 当n>=5时，尽可能多的剪长度为3的绳子，当剩下的长度为4时，把绳子剪为长度为2的两段
// 合理性：当n ≥ 5时，证明2(n - 2)>n并且3(n - 3)>n，即当绳子剩下的长度大于或等于 5 时，剪成长度为 3 或 2 的绳子段更优。
//       同时证明当n ≥ 5时，3(n - 3)≥2(n - 2)，所以应尽可能多剪长度为 3 的绳子段。
function cutString(n) {
    if(n < 2) {
        return 0
    }

    if(n === 2) {
        return 1
    }

    if(n === 3) {
        return 2
    }

   // 计算剪去长度为3的绳子的段数
   const timesOf3 = Math.floor(n / 3);
   // 处理剩余长度为4的情况
   if (n - timesOf3 * 3 === 1) {
       timesOf3--;
   }
   // 计算剪去长度为2的绳子的段数
   const timesOf2 = Math.floor((n - timesOf3 * 3) / 2);
   return Math.pow(3, timesOf3) * Math.pow(2, timesOf2);
}


// 16、请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。
// 例如，输入为 9，其二进制表示为 1001，那么 1 的个数为 2。
// 方法1、可以转为二进制，然后数1的个数

// 方法2、位运算，右移两位，无符号整数 000010101 >>2 = 00000010，10001010 >> 3 = 11110001
// 思路：先判断二进制最右边是不是1，是的话再右移，继续判断最右边是不是1，如此循环，直到判断为0为止

function printOne(n) {
    let count = 0
    // 这有个问题就是负数的话右移会陷入死循环，因为负数右1，最高位补1，算出来的值一直大于0
    while(n) {
        if(n & 1) count++
        n = n >> 1
    }
    return count
}

// 方法2、改进后的做法
// 思路：先把n和1相与，判断最右边是不是1，再把1左移一位得到2，再与n相与判断倒数第二位是不是1，如此循环往复，直到相与为0
function printOne(n) {
    let count = 0
    let flag = 1
    while(flag) {
        if(n & flag) {
            count++
        }

        flag = flag << 1
    }

    return count
}

//方法3
//思路：
// 把一个数减去1，如果最后一位是1，那么最后一位变成了0，其他位不变，想当于最后一位做了取反操作
// 如果最后一位是0，那么该数最右的1位于第m位，减去1时，m位之后的所有数都做了按位取反，第m位变成0，m右边的都变成了1，第m位之前都不变
// 总结上面两种情况就是把一个数减去1，都是把最后一位1变成0，如果右边还有0的话，则所有的0都变成1，而它左边的数都保持不变
// 举个例子二进制数1100，减去1后，变成1011，如果把1100 和1011做与运算，则变成1000，结果刚好是把1100最后一个1变成了0
// 基于这种思路可以依次把一个数的所有为1的位变成0，也就是每一步可以消掉一个1，直到数为0时，代表消掉了所有1，在这个过程中我们可以记录1的个数
function printOne(n) {
    let count = 0
    while(n) {
        count++
        n = (n - 1) & n
    }
    return count
}

// 17、求数值的整数次方
// 方法1
function power(num, exp) {
    const powerCore = (num, exp) => {
        let res = 1
        for(let i = 1; i <= exp; i++) {
            res *= num
        }
        return res
    }
    const isEqual = (num1, num2) => {
        return Math.abs(num1 - num2) < 0.0000001;
    }
    // 判断底数为0且exp < 0
    if (isEqual(num, 0.0) && exp < 0) {
        throw Error('非法参数')
    }
    let res = powerCore(num,Math.abs(exp))
    if(exp < 0) {
        res = 1 / res
    }
    return res
}
// 方法2，通项公式
// 例如求a(n)，当为偶数时，a(n) = a(n/2) * a(n/2)；当为奇数时，a(n)=a((n-1)/2) * a((n-1)/2) * a。
// 通过不断将指数减半并利用位运算判断奇偶，减少乘法次数
function power(num, exp) {
    const powerCore = (num, exp) => {
        if(exp === 0) {
            return 1
        }

        if(exp === 1) {
            return num
        }

        // exp >> 1等价于exp / 2，位运算比较高效
        // 把exp指数拆分，拆分成两个偶数项相乘
        let res = powerCore(num, exp >> 1)
        res *= res
        // 如果exp为奇数的话，不断除2以后会变成1，那么exp & 00000001 为1的话，代表exp为奇数，应该再乘以一个num
        // 以后如果判断一个数为奇数，都可以用exp & 000000001 === 1来判断
        if(exp & 0x1 === 1) {
            res *= num
        }
        return res
    }
    const isEqual = (num1, num2) => {
        return Math.abs(num1 - num2) < 0.0000001;
    }
    // 判断底数为0且exp < 0
    if (isEqual(num, 0.0) && exp < 0) {
        throw Error('非法参数')
    }
    let res = powerCore(num, Math.abs(exp))
    if(exp < 0) {
        res = 1 / res
    }
    return res
}

// 18、打印1到最大的n位十进制数
// 方法1，不推荐，输入n很大的时候，数字会溢出
function printMax(n) {
    if(n <= 0) return
    const max = Math.pow(10, n) - 1
    for(let i = 1; i <= max; i++) {
        console.log(i)
    }
}

// 方法2，使用字符串表示数字，模拟加法操作
function printMax(n) {
    // 字符串模拟加法
    const increment = (number) => {
        let isOverflow = false; // 是否溢出
        let takeOver = 0; // 是否进位
        for (let i = number.length - 1; i >= 0; i--) { // 从最低位开始相加
            let sum = parseInt(number[i]) + takeOver;
            if (i === number.length - 1) { // 最低位加一
                sum++;
            }
            if (sum >= 10) {
                // 当前如果处在最高位，证明溢出了
                if (i === 0) {
                    isOverflow = true;
                    break;
                } else {
                    sum -= 10;
                    // 该进位了
                    takeOver = 1;
                    number[i] = sum.toString();
                }
            } else {
                number[i] = sum.toString();
                break;
            }
        }
        return isOverflow;
    }
    const printNum = (number) => {
        let start = 0;
        // 移动到最高位
        while (number[start] === '0') {
            start++;
        }
        let numStr = '';
        for (let i = start; i < number.length; i++) {
            numStr += number[i];
        }
        console.log(numStr);
    }
    if(n <= 0) return

    const number = new Array(n).fill('0')

    while(!increment(number)) {
        printNum(number)
    }
}

// 19、在o(1)时间内删除链表中的节点
// 顺序遍历找到节点耗时o(n),知道删除节点，把删除节点下一个节点值赋给待删除节点，
// 再将删除节点的next指向删除节点的下一个节点的next，就完成了o(1)，删除
function deleteNode(pHead, pToDeleted) {
    if(!pHead || !pToDeleted) {
        return
    }

    // 要删除的节点不是尾节点
    if(pToDeleted.next) {
        let nextN = pToDeleted.next
        pToDeleted.value = nextN.value
        pToDeleted.next = nextN.next
    } else if(pHead === pToDeleted) { // 要删除头节点
        pHead = null
    } else { // 有多个节点，删除尾节点
        let cur = pHead
        while(cur.next !== pToDeleted) {
            cur = cur.next
        }
        cur.next = null
    }
}

// 20、删除排序链表中的重复节点
function deleteDuplication(pHead) {
    function ListNode(value) {
        this.next = null
        this.value = value
    }

    if(!pHead) {
        return null
    }

    // 因为头节点也有可能是重复节点，所以有可能被删除，
    // 所以应该有个指针指向头节点，以边删除头节点时指向到下一个节点保证链表的连接状态
    let dum = new ListNode(0)
    dum.next = pHead
    // 在遍历链表时，prev 始终指向最后一个不重复节点。
    // 当没有发现重复节点时，prev 会更新为当前节点 cur，表示该节点暂时被认为是不重复的，后续可能会根据情况更新。
    // 当发现重复节点时，prev 保持在最后一个不重复节点的位置，以便将其 next 指针指向重复节点序列之后的节点，确保链表的连贯性。
    let prev = dum
    let cur = pHead
    while(cur) {
        let isDuplication = false
        // 这里用while而不用if是考虑到连续的数字可能会超过2个，所以不断循环，才能让cur的值为连续数字的最后一个
        while(cur.next && cur.next.value === cur.value) {
            isDuplication = true
            cur = cur.next
        }

        if(isDuplication) {
            // 让prev指向连续数字的最后一个的next，则表示删除了前面的连续数字
            prev.next = cur.next
        } else {
            prev = cur
        }

        cur = cur.next
    }

    return dum.next
}

// 21、请实现一个函数用来匹配包含 . 和 * 的正则表达式。模式中的字符 . 表示任意一个字符，而 * 表示它前面的字符可以出现任意次（包含 0 次）。
// 本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串 "aaa" 与模式 "a.a" 和 "ab*ac*a" 匹配，但与 "aa.a" 和 "ab*a" 不匹配。
// 思路：
// 首先，如果模式 p 的长度为 0，那么只有当字符串 s 的长度也为 0 时才匹配成功。
// 然后，判断 s 是否为空，并检查 s 的第一个字符和 p 的第一个字符是否匹配（. 可以匹配任意字符），得到 firstMatch。
// 如果 p 的长度大于等于 2 且第二个字符是 *，有两种情况：
// 一种是认为 * 表示前面的字符出现 0 次，直接跳过 * 及其前面的字符，继续匹配 s 和 p 从第三个字符开始的部分。
// 另一种是 * 表示前面的字符出现 1 次或多次，此时需要 firstMatch 为真，且继续匹配 s 从第二个字符开始和 p 的部分。
// 当 p 的下一个字符不是 * 时，直接比较 s 和 p 的第一个字符是否匹配，并且继续匹配 s 从第二个字符开始和 p 从第二个字符开始的部分
function MathFuc(str, p) {
    if(p.length === 0) {
        return str.length === 0
    }

    // 匹配第一个字符
    let firstMatch = str.length > 0 && (str[0] === p[0] || str[0] === '.')
    if(p.length >= 2 && p[1] === '*') {
        // 当模式的下一个字符是*时，有两种情况
        // 【*】 表示前面的数字出现0次,从第3个字符开始匹配
        //【*】表示前面的数字出现1次或多次，继续匹配
        return MathFuc(str, p.slice(2)) || (firstMatch && MathFuc(str.slice(1), p))
    } else {
        // 当模式的下一个字符不是 '*' 时，直接比较当前字符并继续匹配后续字符
        return firstMatch && MathFuc(str.slice(1), p.slice(1))
    }
}

// 22、表示数值的字符串
// 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
// 例如，字符串 "+100"、"5e2"、"-123"、"3.1416"、"-1E-16" 都表示数值，
// 但 "12e"、"1a3.14"、"1.2.3"、"+-5" 和 "12e+5.4" 不表示数值。
// 分析字符串格式
// 表示数值的字符串遵循特定模式A[.[B]][e|EC]或.B[e|EC]，其中：
//      A为数值的整数部分，可以以+或-开头，后面跟着 0 到 9 的数位串。
//      B为数值的小数部分，是 0 到 9 的数位串，前面不能有正负号。
//      C为数值的指数部分，以e或E开头，后面跟着整数部分。
function isNumber(str) {
    let index = 0

    const scanInteger = (str) => {
        if(str[index] === '-' || str[index] === '+') {
            index++
        }
        return scanUnsignedInteger(str)
    }

    const scanUnsignedInteger = (str) => {
        let start = index
        while(index < str.length && str[index] >= '0' && str[index] <= '9') {
            index++
        }
        return index - start > 0 // 扫描到了数字
    }
    // 1、扫描整数部分
    let isNumber = scanInteger(str)
    
    // 2、扫描小数部分
    if(str[index] === '.') {
        index++
        // 小数可以没有整数部分，所以用或逻辑
        isNumber = scanUnsignedInteger(str) || isNumber
    }
    // 3、扫描指数部分
    if(str[index] === 'e' || str[index] === 'E') {
        index++
        // e前面必须有数字
        isNumber = isNumber && scanInteger(str)
    }


    // 遍历完之后index的值为len
    return isNumber && str.length === index
}

// 23、调整数组顺序使奇数位于偶数前面
// 输入一个整数数组，要求调整数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
// 例如，输入数组[1,2,3,4,5,6,7]，调整后的数组可以是[1,3,5,7,2,4,6]
// 方法1、双指针
function reorderOddEven(arr) {
    if (!arr || arr.length === 0) {
        return arr; // 如果数组为空，直接返回
    }
    
    let p1 = 0; // 指向数组的开始
    let p2 = arr.length - 1; // 指向数组的结束
    
    while (p1 < p2) {
        // 找到左边的偶数
        while (arr[p1] % 2 === 1 && p1 < p2) {
            p1++; // 移动指针到下一个奇数
        }
        
        // 找到右边的奇数
        while (arr[p2] % 2 === 0 && p1 < p2) {
            p2--; // 移动指针到下一个偶数
        }
        
        // 交换偶数和奇数
        if (p1 < p2) {
            [arr[p1], arr[p2]] = [arr[p2], arr[p1]];
        }
    }
    
    return arr; // 返回调整后的数组
}

// 方法2、通用思路，可以根据不同条件调整元素
// 其中condition写具体的条件
// 如本题的condition为:
// function isEven(num) {
//     return num % 2 === 0;
// }
function reOrder(arr, condition) {
    if (!arr || arr.length === 0) {
        return arr;
    }
    let left = 0;
    let right = arr.length - 1;
    while (left < right) {
        while (left < right && condition(arr[left])) {
            left++;
        }
        while (left < right &&!condition(arr[right])) {
            right--;
        }
        if (left < right) {
            let temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
        }
    }
    return arr;
}

// 24、链表中倒数第k个节点
// 输入一个链表，输出该链表中倒数第 k 个节点。
// 例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。那么倒数第 3 个节点是值为 4 的节点。
// 方法1、快慢指针， 先让快指针走k步，然后快慢指针一起走，直到快指针走到末尾，慢指针就在倒数第k个
function findK(pHead, k) {
    if(!pHead || k <= 0) {
        return null
    }

    let p1 = pHead // 快指针
    let p2 = pHead // 慢指针
    while(k) {
        if(!p1) { // 如果p1到达链表末尾，代表k大于链表长度
            return null
        }
        p1 = p1.next
        k--
    }
    while(p1) {
        p1 = p1.next
        p2 = p2.next
    }
    return p2
}
// 方法2.遍历链表存在数组里，这个应该没有分，不写了

// 25、链表中环的入口节点
// 方法1
// 典型的快慢指针
//      首先使用快慢指针判断链表是否有环。
//      快指针 fast 每次走两步，慢指针 slow 每次走一步，如果它们相遇，说明有环。
//      相遇后，将慢指针重新指向头节点，快指针和慢指针都改为每次走一步，再次相遇的节点就是环的入口节点。
function circlePoint(pHead) {
    if(!pHead) {
        return null
    }
    let fast = pHead
    let slow = pHead
    // 快指针一次走2步，慢指针一次走1步
    while(fast && fast.next) {
        fast = fast.next.next
        slow = slow.next
        // 相遇时说明有环
        if(slow === fast) {
            // 将慢指针重新指向头节点
            slow = pHead
            // 快指针和慢指针都改为每次走一步，再次相遇的节点就是环的入口节点
            // 这是因为当快慢指针相遇时，设从链表头节点到环入口的距离为 a，环的长度为 b，
            // 快慢指针相遇时慢指针走了 x 步，快指针走了 2x 步，且快指针比慢指针多走了 n 个环的长度（2x = x + nb），可得 x = nb。
            // 将慢指针移到头节点，再走 a 步，而快指针从相遇点走 a 步也会到达环入口，所以再次相遇点就是环入口。
            while(slow !== fast) {
                slow = slow.next
                fast = fast.next
            }
            return slow
        }
    }
    return null
}
// 方法2.标记法
function circlePoint(pHead) {
    if(!pHead) {
        return null
    }
    let cur = pHead
    while(cur) {
        if(cur.visited) {
            return cur
        }
        cur.visited = true
        cur = cur.next
    }
    return null
}

// 25、反转链表
// 迭代法
function reverseList(pHead) {
    let prev = null // 新的头节点
    let cur = pHead // 指向当前节点，初始化指向头节点
    while(cur) {
        let temp = cur.next // 保存当前节点的下一个节点
        cur.next = prev // 当前节点反转指向prev
        prev.next = cur // prev作为头节点必须指向目前的第一个节点，所以要指向cur
        cur = temp // 当前节点已经处理完了，所以要到下一个节点了
    }

    return prev
}

// 递归法
// 空间复杂度为o(1)
// 递归函数 reverseList 接收当前节点 pHead 和前一个节点 prev 作为参数。
// 递归的终止条件是当 head 为 null，返回 prev。
// 递归过程中，先存储下一个节点，将当前节点的 next 指向前一个节点，然后递归调用函数处理下一个节点。
function reverseList(pHead, prev = null) {
    if(!pHead) {
        return prev
    }
    let next = pHead.next
    pHead.next = prev
    return reverseList(next, pHead)
}

// 26、合并两个排序链表
// 输入两个递增排序的链表，将它们合并为一个递增排序的链表。
// 例如，输入链表 1 -> 3 -> 5 和 2 -> 4 -> 6，合并后的链表为 1 -> 2 -> 3 -> 4 -> 5 -> 6
function combineTwoList(l1, l2) {
    let dum = new ListNode(0); // 创建一个虚拟头节点
    let cur = dum; // 当前节点指针
    while (l1 && l2) { // 当两个链表都不为空时
        if (l1.value < l2.value) { // 比较当前节点的值
            cur.next = l1; // 将较小的节点链接到结果链表
            l1 = l1.next; // 移动到下一个节点
        } else {
            cur.next = l2; // 将较小的节点链接到结果链表
            l2 = l2.next; // 移动到下一个节点
        }
        cur = cur.next; // 移动当前节点指针
    }
    if (l1) { // 如果 l1 还有剩余节点
        cur.next = l1; // 直接链接剩余节点
    }
    if (l2) { // 如果 l2 还有剩余节点
        cur.next = l2; // 直接链接剩余节点
    }

    return dum.next; // 返回合并后的链表，跳过虚拟头节点
}

// 27、对一组年龄进行排序
function sortAges(ages) {
    if(!ages || ages.length <= 0) {
        return;
    }

    const oldestAge = 99;
    let timesOfAge = new Array(oldestAge + 1).fill(0);
    // 记录每个年龄出现的次数
    for(let i = 0; i < ages.length; i++) {
        let age = ages[i];
        if(age < 0 || age > oldestAge) {
            throw new Error('age out of range');
        }
        timesOfAge[age]++;
    }

    // 依次取出年龄，按次数放回原数组
    for(let i = 0, index = 0; i <= oldestAge; i++) {
        for(let j = 0; j < timesOfAge[i]; j++) {
            ages[index] = i;
            index++
        }
    }
    return ages;
}

// 28、树的子结构
// 输入两棵二叉树A和B，判断B是不是A的子结构
// 思路： 
//  首先在树A中找到和树B的根节点值相同的节点
//  然后判断以该节点为根的子树是否和树B的结构相同。
function isSubtree(root1, root2) {
    const hasSubtreeCore = (root1, root2) => {
        if(!root2) { // root2已遍历完成，代表到现在为止都匹配
            return true
        }
        if(!root1) { // root1已经遍历完成，但是root2还有节点，代表root1不可能包含了root2
            return false
        }

        if(root1.value !== root2.value) {
            return false
        }

        // 如果当前节点值相同，就去判断左子树和右子树分别是不是子结构
        return hasSubtreeCore(root1.left, root2.left) && hasSubtreeCore(root1.right, root2.right)
    }
    if(!root1 || !root2) {
        return false
    }

    // 如果root1和root2的value相同，则判断是不是子结构
    // 否则分别去递归查看root1的左右子树，直到找到节点值和root2的根节点值相同的，然后再调用hasSubtreeCore判断子结构
    return hasSubtreeCore(root1, root2) || isSubtree(root1.left, root2) || isSubtree(root1.right, root2)
}


// 29、二叉树的镜像
// 思路：构建二叉树的镜像就是遍历过程中交换非叶节点的左右值
function MirrorTree(root) {
    if(!root) {
        return null
    }

    // 交换左右子节点
    let temp = root.left
    root.left = root.right
    root.right = temp

    // 递归处理左右子树
    MirrorTree(root.left)
    MirrorTree(root.right)
    return root
}


// 30、对称二叉树
// 请实现一个函数，用来判断一棵二叉树是否对称。如果一棵二叉树和它的镜像一样，那么它是对称的。例如，下面这棵二叉树是对称的：
//     1
//    / \
//   2   2
//  / \ / \
// 3  4 4  3
// 思路：对于对称二叉树，比如前序遍历，是先左后右，如果设计一个遍历方式，先右后左，对两种方式同时遍历，如果过程中对应的节点值都相同，代表就是对称二叉树
// 可以使用递归的方式来判断二叉树是否对称。
//      设计一个辅助函数，比较二叉树的左右子树是否对称。
//      对于两个节点 node1 和 node2，
//          如果它们都为 null，则认为对称；
//          如果其中一个为 null 而另一个不为 null，则不对称；
//          如果它们的值不相等，则不对称；
//          如果它们的值相等，则递归比较 node1 的左子树和 node2 的右子树、node1 的右子树和 node2 的左子树是否对称。
function isSymmetric(root) {
    const isSymmetricCore = (node1, node2) => {
        if(node1 === null && node2 === null) {
            return true
        }

        if(node1 !== null || node2 !== null) {
            return false
        }

        if(node1.value !== node2.value) {
            return false
        }

        return isSymmetricCore(node1.left, node2.right) && isSymmetricCore(node1.right , node2.left)
    }
    if(!root) return true

    return isSymmetricCore(root.left, root.right)
}

// 迭代版本：通过队列来存储节点，比较节点的左右子节点是否对称。具体做法是将左右子节点成对地加入队列，然后依次取出进行比较。
function isSymmetricIterative(root) {
    if (root === null) {
        return true;
    }
    let queue = [];
    queue.push(root.left);
    queue.push(root.right);
    while (queue.length > 0) {
        let node1 = queue.shift();
        let node2 = queue.shift();
        if (node1 === null && node2 === null) {
            continue;
        }
        if (node1 === null || node2 === null) {
            return false;
        }
        if (node1.val!== node2.val) {
            return false;
        }
        queue.push(node1.left);
        queue.push(node2.right);
        queue.push(node1.right);
        queue.push(node2.left);
    }
    return true;
}


// 31、顺时针打印矩阵
// 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每个数字
//  [
//     [1, 2, 3],
//     [4, 5, 6],
//     [7, 8, 9]
//   ]
function printMatrixClockwise(matrix) {
    if(matrix === null || matrix.length === 0 || matrix[0].length === 0) {
        return []
    }
    let res = []
    let left = 0
    let right = matrix[0].length - 1
    let top = 0
    let bottom = matrix.length - 1
    while(top <= bottom && left <= right) {
        // 从左到右打印
        for(let i = left; i <= right; i++) {
            res.push(matrix[top][i])
        }
        top++

        // 从上到下打印
        if(top <= bottom) {
            for(let i = top; i <= bottom; i++) {
                res.push(matrix[i][right])
            }
            right--
        }
       

        // 从右往左打印
        if(top <= bottom && left <= right) {
            for(let i = right; i >= left; i--) {
                res.push(matrix[bottom][i])
            }
            bottom--
        }

        // 从下往上打印
        if(top <= bottom && left <= right) {
            for(let i = bottom; i >= top; i--) {
                res.push(matrix[i][left])
            }
            left++
        }
    }
    return res
}
 
// 32、包含min函数的栈
// 定义栈的数据结构，在该栈中实现一个能够得到栈的最小元素的 min 函数，要求 min、push 及 pop 操作的时间复杂度都是 o(1)
class MinStack {
    constructor() {
        this.stack = [] // 储存元素
        this.minStack = [] // 储存栈的最小元素
    }

    push(value) {
        this.stack.push(value)
        if(this.minStack.length === 0 || value <= this.minStack[this.minStack.length - 1]) {
            this.minStack.push(value)
        }
    }

    pop() {
        if(this.stack.pop() === this.minStack[this.minStack.length - 1]) {
            this.minStack.pop()
        }
    }

    min() {
        if(this.minStack.length === 0) {
            return null
        }
        return this.minStack[this.minStack.length - 1]
    }
}

// 33、栈的压入、弹出序列
// 输入两个整数序列，第一个序列表示栈的压入顺序，判断第二个序列是否为该栈的弹出顺序。
// 例如，压入序列为 [1, 2, 3, 4, 5]，弹出序列为 [4, 5, 3, 2, 1] 是合理的，
// 因为可以按以下操作得到：将 1、2、3、4 依次压入栈，然后弹出 4，压入 5，再依次弹出 5、3、2、1；而弹出序列为 [4, 3, 5, 1, 2] 则不合理
// 思路：设计一个辅助栈， 规律： 
//      如果下一个弹出的数字刚好是栈顶数字，则弹出；
//      如果下一个弹出的数字不在栈顶，则把压栈序列还没入栈道数字入栈，直到把下一个需要弹出的数字压入栈顶为止；
//      如果所有数字都入栈了，但是还没有找到下一个应该弹出的数字，代表该序列不可能是一个弹出序列
// 方法1
function isPopOrder(pushList, popList) {
    if (pushList === null || popList === null || pushList.length!== popList.length) return false;
    let stack = [];
    let pushIndex = 0; // 指向入栈序列的索引
    for (let popIndex = 0; popIndex < popList.length; popIndex++) {
        // 如果栈顶元素不等于下一个应该弹出的数字，则应该继续入栈
        while (stack.length === 0 || stack[stack.length - 1]!== popList[popIndex]) {
            // 如果所有数字都入栈了，代表该序列不可能是一个弹出序列
            if (pushIndex === pushList.length) {
                return false;
            }
            pushIndex++
            stack.push(pushList[pushIndex]);
        }

        // 栈顶元素不等于下一个应该弹出的数字，弹出
        stack.pop();
    }
    return true;
}
// 方法2 这个比较好理解
function isPopOrder(pushList, popList) {
    if(!popList || !pushList || pushList.length !== popList.length) {
        return false
    }

    let stack = []
    let p = 0 // 指向弹出序列的索引
    // 模拟入栈出栈
    for(let num of pushList) {
        stack.push(num)
        // 如果栈顶元素等于下一个应该弹出的数字，则弹出，更新弹出序列的索引
        while(stack.length > 0 && stack[stack.length - 1] === popList[p]) {
            stack.pop()
            p++
        }
    }

    // 如果结果是正确的，那么弹出序列的索引应该已经增长为弹出序列的长度
    return p === popList.length
}

// 34、从上到下打印二叉树
//     1
//    / \
//   2   3
//  / \ / \
// 4  5 6  7
//a、 不分行从上到下打印二叉树
// 本质就是层序遍历，也就是广度优先搜索
function printFromTopToBottom(root) {
    if(root === null) {
        return 
    }
    let queue = [] // 层序遍历通常是用队列实现的
    queue.push(root)
    while(queue.length > 0) {
        let node = queue.shift()
        console.log(node.value)
        if(node.left) {
            queue.push(node.left)
        }

        if(node.right) {
            queue.push(node.right)
        }
    }
}
// b、分行从上到下打印二叉树
// 本质就是层序遍历，也就是广度优先搜索
function printTreeInLines(root) {
    if(root === null) {
        return 
    }
    let queue = [] // 层序遍历通常是用队列实现的
    queue.push(root)
    let toPrint = 1 //当前层未打印节点数
    let nextLevel = 0 // 下一层的节点数
    while(queue.length > 0) {
        let node = queue.shift()
        console.log(node.value)
        if(node.left) {
            queue.push(node.left)
            nextLevel++
        }

        if(node.right) {
            queue.push(node.right)
            nextLevel++
        }

        toPrint--
        // 当前层节点已经打印结束，此时nextLevel的值是下一层所有节点的个数
        // 更新toPrint和nextLevel
        if(toPrint === 0) { 
            console.log(' ')
            toPrint = nextLevel
            nextLevel = 0
        }
    }
}

// c、之字形打印二叉树
// 请实现一个函数按照之字形顺序打印二叉树，
// 即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
// 思路：
//      需要两个栈。打印奇数层时，先将根节点入栈 1，然后从栈 1 弹出节点打印，将其左右子节点按先左后右顺序入栈 2；
//      打印偶数层时，从栈 2 弹出节点打印，将其左右子节点按先右后左顺序入栈 1。如此交替，直到两个栈都为空
function printTreeInZigzag(root) {
    if(root === null) {
        return 
    }
    let stack1 = [] // 存储奇数层
    let stack2 = [] // 存储偶数层
    stack1.push(root)
    let cur = true // 判断奇数层还是偶数层， true奇数 false偶数
    let toPrint = 1 //当前层未打印节点数
    let nextLevel = 0 // 下一层的节点数
    while(stack1.length > 0 || stack2.length > 0) {
        let node
        if(cur) { //奇数层
            node = stack1.shift()
            console.log(node.value)
            if(node.right) {
                stack2.push(node.right)
                nextLevel++
            }

            if(node.left) {
                stack2.push(node.left)
                nextLevel++
            }
        } else {
            node = stack2.shift()
            console.log(node.value)
            if(node.left) {
                stack1.push(node.left)
                nextLevel++
            }
            if(node.right) {
                stack1.push(node.right)
                nextLevel++
            }
        }

        toPrint--
        if(toPrint === 0) { 
            console.log(' '); // 打印空行
            toPrint = nextLevel; // 更新当前层未打印节点数
            nextLevel = 0; // 重置下一层节点数
            cur = !cur; // 切换层级
        }
    }
}

// 35、二叉搜索树的后续遍历序列
// 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。假设输入的数组的任意两个数字都互不相同。
//        8
//       /  \
//      6    10
//     / \   / \
//    5   7 9  11
// 例如，输入数组 {5,7,6,9,11,10,8}，则返回true，因为这个整数序列是例子中 二叉搜索树的后序遍历结果。
//      如果输入的数组是 {7,4,6,5}，则由于没有哪棵二叉搜索树的后序遍历结果是这个序列，因此返回false
// 思路： 后续遍历的特征是左子树、右子树、根节点，二叉搜索树中左子树节点的值小于根节点的值，右子树节点的值大于根节点的值
//     a、 确定根节点：数组的最后一个元素是根节点的值。
//     b、 划分左右子树：在数组中找到第一个大于根节点的值的位置，该位置左边的元素构成左子树的节点值，右边的元素构成右子树的节点值。
//     c、 递归验证子树：分别对左子树和右子树的数组进行递归验证，判断它们是否满足二叉搜索树的后序遍历特征
function verifySequenceOfBST(list) {
    let len = list.length
    if(!list || len === 0) {
        return false
    }

    let rootValue = list[len - 1]
    
    // 在二叉搜索树中左子树节点的值小于右子树节点的值
    let i = 0
    for(; i < len - 1; i++) {
        if(list[i] > rootValue) {
            break
        }
    }

    // 在二叉搜索树中右子树节点的值大于根节点的值
    let j = i
    for(; j < len - 1; j++) {
        if(list[j] < rootValue) {
            return false
        }
    }

    // 判断左子树是不是二叉搜索树
    let left = true
    if(i > 0) {
        left = verifySequenceOfBST(list.slice(0, i))
    }

    // 判断右子树是不是二叉搜索树
    let right = true
    if(i < len - 1) {
        right = verifySequenceOfBST(list.slice(i, len - 1))
    }
    
    return left && right
}
// 反思： 如果面试题要求处理一颗二叉树的遍历序列，可以找到二叉树的根节点，再基于根节点把遍历序列拆分成左右子树对应的序列，再递归地处理这两个子序列
// 重建二叉树就属于这种思路


// 36、二叉树中和为某一值的路径
// 输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径
// 看起来像是回溯算法
// 在三种遍历方式中，只有前序遍历是从根节点开始的，所以我们在问题中要讨论的是前序遍历
function printPath(root, val) {
    const pathCore = (node, val, path, sum) => {
        if(node === null) {
            return
        }
        // 根
        sum += node.val
        path.push(node.val)

        // 如果累积和相等则找到了
        if(val === sum && node.left === null && node.right === null) {
            console.log(path)
        }

        // 左
        pathCore(node.left, val, path, sum)
        // 右
        pathCore(node.right, val, path, sum)
        path.pop() // 回溯
    }
    if(root === null) {
        return
    }

    let path = []
    pathCore(root, val, path, 0)
}

// 37、复杂链表的复制
// 请实现函数ComplexListNode*Clone(ComplexListNode* pHead)，复制一个复杂链表。
// 在复杂链表中，每个节点除了有一个m_pNext指针指向下一个节点，还有一个m_pSibling指针指向链表中的任意节点或者null
// 思路1:常规思路是先复制每个节点然后连起来，然后再为每个节点的m_pSibling赋值，每为一个节赋值都要遍历一遍，这样的时间复杂度是o(n2)，显然不行
// 思路2: 第一步还是复制每个节点并连起来，在复制的同时用哈希表记录每个节点的m_pSibling，然后在根据哈希表的对应关系赋值，时间复杂度是o(n),但也引入了额外的空间o(n)
// 思路3: 怎么才能不使用额外内存解决问题呢，把时间复杂度保持在o(n)
//      分解问题，把复杂问题分解成小问题分别解决，也就是所谓的分治法
//      第一步：复制链表的每个节点N为N`，然后把N`连在N后面，如：A->A`->B->B`->C->C`->D->D`->E->E`
//      第二步：根据原节点设置复制出来的节点的m_pSibling，例如：如果原节点A指向C，那么A`指向C`
//      第三步：把整个链表拆分成两个链表，奇数位置的节点连起来就是原链表，偶数位置的节点连起来就是新的链表
class ComplexListNode {
    constructor(val) {
        this.val = val;
        this.m_pNext = null;
        this.m_pSibling = null;
    }
}

function Clone(head) {
    const cloneNodes = (head) => {
        let cur = head
        while(cur)  {
            let newNode = new ComplexListNode(cur.val)
            newNode.m_pNext = cur.m_pNext
            cur.m_pNext = newNode
            cur = newNode.m_pNext
        }
    }
    
    const connectSiblingNodes = (head) => {
        let cur = head
        while(cur) {
            if(cur.m_pSibling) {
                cur.m_pNext.m_pSibling = cur.m_pSibling.m_pNext
            }
            cur = cur.m_pNext.m_pNext
        }
    }

    const reconnectNodes = (head) => {
        let orgHead = head
        let newHead = head.m_pNext
        let orgCur = orgHead
        let newCur = newHead

        while(orgCur) {
            orgCur.m_pNext = orgCur.m_pNext.m_pNext
            newCur.m_pNext = newCur.m_pNext ? newCur.m_pNext.m_pNext : null
            orgCur = orgCur.m_pNext
            newCur = newCur.m_pNext
        }
        return newHead
    }

    if(head === null) {
        return null
    }

    // 第一步：复制节点
    cloneNodes(head)
    // 第二步：连接m_pSibling
    connectSiblingNodes(head)

    // 第三步： 拆分链表
    return reconnectNodes(head)
}

// 38、二叉搜索树与双向链表
// 将二叉搜索树转换成一个排序的双向链表，要求不能创建任何新的节点，只能调整树中节点指针的指向。
//        10
//       /  \
//      6    14
//     / \   / \
//    4   8 12  16
// 转换结果 4 -> 6 -> 8 -> 10 -> 12 -> 14 -> 16,其中->代表双向的
// 例如，将上面的二叉搜索树转换成排序的双向链表，使链表中的节点依次为 1, 2, 3, 4, 5, 6, 7, 8, 9, 10。
// 思路：乍看起来要中序遍历二叉搜索树，因为中序遍历，结果是单调递增的
//      利用二叉搜索树的特性，中序遍历可以得到一个有序的节点序列。
//      在中序遍历过程中，调整节点的指针，将左子节点作为前驱指针，右子节点作为后继指针。
function Convert(root) {
    const covertCore = (node) => {
        if(node === null) {
            return
        }

        // 遵循中序遍历，左根右

        // 首先递归转换左子树
        covertCore(node.left)

        // 将当前节点的左指针指向之前转换得到的最后一个节点
        node.left = lastListNode

        // 若之前有节点，将其右指针指向当前节点
        if(lastListNode) {
            lastListNode.right = node
        }

        // lastListNode始终指向链表末尾，所以要移动到node处
        lastListNode = node

        // 递归转换右子树
        covertCore(node.right)
    }

    let lastListNode = null // 始终会指向链表末尾
    covertCore(root)
    let head = lastListNode
    // 在转换操作完成后lastListNode会指向链表末尾，所以应该左右到链表头部，得到头节点
    while(head && head.left) {
        head = head.left
    }
    return head
}

// 39、序列化二叉树
// 实现两个函数，分别用来序列化和反序列化二叉树。
// 序列化是将二叉树按照某种遍历顺序转化为一个字符串，以便于存储和传输；
// 反序列化是将序列化后的字符串恢复为原来的二叉树
//        10
//       /  \
//      6    14
//     / \   / \
//    4   8 12  16
// 思路1：
//      要确定一个二叉树，需要前序遍历和中序遍历，或者中序遍历和后序遍历
//      按照这个思路下来：
//          二叉树的序列化方法，那可以序列化中序遍历和前序遍历
//          二叉树的反序列化方法，就是以前序遍历和中序遍历重建二叉树   
//      这个方法限制有两个 1、二叉树不能有重复的节点 2、只有当两个序列中所有数据都读出后才能反序列化，效率比较慢
// 思路2：
//      实际上二叉树的序列化如果是从根节点开始的，那么相应的反序列化实际上在读出根节点的数值时就可以开始了
//      序列化：因此可以使用前序遍历序列化二叉树，在遍历二叉树的时候，如果碰到null，可以使用一个特殊字符表示如#
//      反序列化： 
//          将序列化得到的字符串按照分隔符（如 ,）拆分成节点值列表。
//          利用递归函数，根据节点值列表的顺序，依次构建二叉树。
//          当遇到特殊字符 # 时，表示该位置是一个空节点
// 序列化，前序遍历
function serialize(root) {
    let res = ''
    const preOrderCore = (root) => {
        if(root === null) {
            res += '#,'
            return
        }
        res += root.val + ','
        preOrderCore(root.left)
        preOrderCore(root.right)
    }

    preOrderCore(root)
    return res
}
// 反序列化
function deSerialize(str) {
    const nodes = str.split(',')
    const index = 0 // 用于遍历nodes数组
    const buildTree = () => {
        if(nodes[index] === '#') {
            index++
            return null
        }

        let node = new BinaryTreeNode(nodes[index].val)
        index++

        // 前序遍历的左子树和右子树节点挨着，所以遍历完左子树，剩下的节点都是右子树的节点，所以这样顺序写没问题
        node.left = buildTree()
        node.right = buildTree()
        return node
    }

    return buildTree()
}

// 40、字符串的排列
// 输入一个字符串，打印出该字符串中字符的所有排列,如abc，打印abc、acb、bac、bca、cab、cba
// 思路：典型的回溯法
//        两个子问题：第一步、选取一个字符
//                  第二步、处理其他字符
//                  第二步右可以拆分成两个子问题递归处理
function Permutation(str) {
    if(!str) {
        return
    }

    const permuatCore = (currentStr, res) => {
        if(currentStr.length === 0) {
            console.log(res)
            return
        }
        for(let i = 0; i < currentStr.length; i++) {
            // 第一个子问题：选取一个字符
            // 更新res
            const newRes = res + currentStr[i]; // 创建新的res

            // 第二个子问题：递归处理其他字符
            // 数组中去除i处的元素
            const newCurrentStr = currentStr.slice(0, i).concat(currentStr.slice(i + 1))
            // 用没有i元素的数组继续递归，直到数组值为0就可以打印出结果了
            permuatCore(newCurrentStr, newRes)
        }
    }

    permuatCore(str.split(''), '') // 将字符串转换为数组
}

// 42、求字符串的中的字符构成的所有组合，如abc，结果是['', 'a', 'ab', 'abc', 'ac', 'b', 'bc', 'c']
function generateCombinations(str) {
    const result = [];
    const combinationHelper = (start, currentCombination) => {
        // 将当前组合添加到结果中
        result.push(currentCombination);
        
        // 选择当前字符并继续生成组合
        for (let i = start; i < str.length; i++) {
            // 第一个子问题：选取一个字符
            const cur = currentCombination + str[i]

            // 第二个子问题：处理其他字符
            // 选择当前字符并继续递归
            combinationHelper(i + 1, cur);
        }
    };

    combinationHelper(0, ''); // 从第一个字符开始
    return result;
}

// 43、数组中超过一半的数字
// 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字
// 思路1: 使用哈希表来统计每个值出现的次数，然后找出
// 思路2: 排序，然后找出
// 思路3:  基于Partition，时间复杂度为o(n)
//    a、 数组中有一个数长度超过一半，也就是数组排序后，数组中间数字必定是这个数字
//    b、 受快速排序算法的启发，通过随机选择一个数字并调整数组中数字的顺序，使得比选中的数字小的数字都排在它的左边，比选中的数字大的数字都排在它的右边。
//        如果选中数字的下标刚好是数组长度的一半，那么这个数字就是中位数；
//        如果下标大于一半，中位数在左边部分的数组中；
//        如果下标小于一半，中位数在右边部分的数组中
function MoreThanHalfNum(numbers) {
    if(!numbers || numbers.length === 0) {
        return null
    }
    let len = numbers.length

    // 检查是否大于一半
    const checkMoreThanHalf = (numbers, num, len) => {
        let times = 0
        for(let i = 0; i < len; i++) {
            if(numbers[i] === num) {
                times++
            }
        }

        return times * 2 > len
    }

    // 这个方法是快速排序最重要的部分，时间复杂度为o(n)
    // 通过选择一个枢轴元素，将数组分为两部分，使得左边的元素都小于等于枢轴，右边的元素都大于等于枢轴
    const Partition = (numbers, start, end) => {
        let pivot = numbers[end]; // 选择最后一个元素作为基准（pivot）
        let i = start - 1; // i 用于跟踪小于或等于 pivot 的元素的最后一个索引

        for (let j = start; j < end; j++) { // 遍历从 start 到 end - 1 的元素
            if (numbers[j] <= pivot) { // 如果当前元素小于或等于 pivot
                i++; // 增加 i 的值
                [numbers[i], numbers[j]] = [numbers[j], numbers[i]]; // 交换元素
            }
        }
        // 将 pivot 放到正确的位置
        [numbers[i + 1], numbers[end]] = [numbers[end], numbers[i + 1]];
        return i + 1; // 返回 pivot 的最终位置
    } 

    let mid = len >> 1
    let start = 0
    let end = len - 1
    let index = Partition(numbers,  start, end)
    // 其实就是找第mid大的数字，找到之后左边的数比mid处小，右边大
    while(index !== mid) {
        // 如果下标大于一半，中位数在左边部分的数组中
        if(index > mid) {
            end = index - 1
            index = Partition(numbers, start, end)
        } else { // 如果下标小于一半，中位数在右边部分的数组中
            start = index + 1
            index = Partition(numbers, start, end)
        }
    }

    // 如果下标等于一半，那么这个数字就是中位数
    let num = numbers[index]

    // 检查是否是超过一半次数
    if(checkMoreThanHalf(numbers, num, len)) {
        return num
    }

    return null
}

// 思路4: 数组中一个数次数超过数组长度的一半，也就是说它出现的次数比其他数字出现的次数和还多
//    统计数字和次数：
//          在遍历数组的时候保存两个值，一个是数组中的一个数字，另一个是次数。
//          当遍历到下一个数字时，如果与保存的数字相同，次数加 1；如果不同，次数减 1。
//    确定目标数字：
//          如果次数为零，就保存下一个数字，并把次数设为 1。
//          由于要找的数字出现的次数比其他所有数字出现的次数之和还要多，所以最后一次把次数设为 1 时对应的数字就是要找的数字。
//          例如:
//              如果数组是 [1, 2, 1, 1, 3, 1, 1]，在遍历过程中，times 会随着数字的不同而增减。
//              当遇到不同数字时，times 减 1；遇到相同数字时，times 加 1。
//              当遇到某个数字使得 times 变为 0 时，就说明之前的数字出现次数已经与其他数字出现次数平衡，此时更新 result 为当前数字，
//              继续下一个数字的处理。最后，result 存储的就是出现次数超过一半的数字,因为它无法跟其他数字达到次数平衡，times不可能减到0
// 上面这种思想也叫摩尔投票算法
function MoreThanHalfNum(numbers) {
    if (!numbers || numbers.length === 0) {
        return null;
    }
    let len = numbers.length;

    // 检查是否大于一半
    const checkMoreThanHalf = (numbers, num, len) => {
        let times = 0;
        for (let i = 0; i < len; i++) {
            if (numbers[i] === num) {
                times++;
            }
        }
        return times * 2 > len;
    };

    let res = numbers[0]; // 初始化结果为第一个元素
    let times = 1; // 初始化计数为 1

    for (let i = 1; i < len; i++) {
        if (numbers[i] === res) {
            times++; // 如果当前元素等于结果，增加计数
        } else {
            times--; // 否则减少计数
        }

        // 如果计数为 0，更新结果
        if (times === 0) {
            res = numbers[i]; // 更新结果为当前元素
            times = 1; // 重置计数为 1
        }
    }

    // 检查是否是超过一半次数
    if (checkMoreThanHalf(numbers, res, len)) {
        return res;
    }

    return null;
}


// 44、实现快速排序
// 这个方法是快速排序最重要的部分，时间复杂度为o(n)
// 通过选择一个枢轴元素，将数组分为两部分，使得左边的元素都小于等于枢轴，右边的元素都大于等于枢轴
const partition = (numbers, start, end) => {
    let pivot = numbers[end]; // 选择最后一个元素作为基准（pivot）
    let i = start - 1; // i 用于跟踪小于或等于 pivot 的元素的最后一个索引

    for (let j = start; j < end; j++) { // 遍历从 start 到 end - 1 的元素
        if (numbers[j] <= pivot) { // 如果当前元素小于或等于 pivot
            i++; // 增加 i 的值
            [numbers[i], numbers[j]] = [numbers[j], numbers[i]]; // 交换元素
        }
    }
    // 将 pivot 放到正确的位置
    [numbers[i + 1], numbers[end]] = [numbers[end], numbers[i + 1]];
    return i + 1; // 返回 pivot 的最终位置
} 

// 方法1: 递归版本
function quickSort(numbers, start, end) {
    if (start < end) { // 递归的终止条件
        // 调用 partition 函数，获取基准元素的最终位置
        const pivotIndex = partition(numbers, start, end);
        // 递归排序基准元素左侧的部分
        quickSort(numbers, start, pivotIndex - 1);
        // 递归排序基准元素右侧的部分
        quickSort(numbers, pivotIndex + 1, end);
    }
}

// 方法2:迭代版本
function quickSort(numbers, start = 0, end = numbers.length - 1) {
    const stack = []; // 创建一个栈来存储待排序的区间
    // 将初始区间压入栈中
    stack.push({ start, end });

    while (stack.length > 0) {
        // 从栈中弹出一个区间
        const { start, end } = stack.pop();

        if (start < end) {
            // 调用分区函数，获取基准元素的最终位置
            const pivotIndex = partition(numbers, start, end);

            // 将基准元素左侧的部分压入栈中
            stack.push({ start, end: pivotIndex - 1 });
            // 将基准元素右侧的部分压入栈中
            stack.push({ start: pivotIndex + 1, end });
        }
    }
}

// 45、找出数组中出现次数最多的一个数， 不关心是否有多个数字出现相同的最大次数
function find(numbers) {
    let length = numbers.length
    let result = numbers[0];
    let times = 1;
    for (let i = 1; i < len; i++) {
        if (numbers[i] === res) {
            times++; // 如果当前元素等于结果，增加计数
        } else {
            times--; // 否则减少计数
        }

        // 如果计数为 0，更新结果
        if (times === 0) {
            res = numbers[i]; // 更新结果为当前元素
            times = 1; // 重置计数为 1
        }
    }
    return result;
}

// 46、最小的k个数
// 输入 n 个整数，找出其中最小的 k 个数。
// 例如，输入 4, 5, 1, 6, 2, 7, 3, 8 这 8 个数字，则最小的 4 个数字是 1, 2, 3, 4
// 思路1: 排序, 最垃圾的方法，没分的
function getLeastNumbersBySort(numbers, k) {
    if (k <= 0 || k > numbers.length) {
        return [];
    }
    numbers.sort((a,b) => a - b)
    return numbers.slice(0, k);
}
// 思路2: 快排,跟上面的没太大区别
function getLeastNumbersBySort(numbers, k) {
    if (k <= 0 || k > numbers.length) {
        return [];
    }

    quickSort(numbers, 0, numbers.length - 1)
    return numbers.slice(0, k);
}
// 思路3: 基于基于Partition，时间复杂度为o(n)，在允许修改输入的数组的情况下, 推荐
//      如果基于数组的第k个数字调整，使得比第k个数字小的数组都位于数组左边，比第k个数字大的数字都位于数组右边，
//      这样位于数组左边的k个数字就是最小的k个数字
function getLeastNumbersBySort(numbers, k) {
    const partition = (numbers, start, end) => {
        let pivot = numbers[end]; // 选择最后一个元素作为基准（pivot）
        let i = start - 1; // i 用于跟踪小于或等于 pivot 的元素的最后一个索引
    
        for (let j = start; j < end; j++) { // 遍历从 start 到 end - 1 的元素
            if (numbers[j] <= pivot) { // 如果当前元素小于或等于 pivot
                i++; // 增加 i 的值
                [numbers[i], numbers[j]] = [numbers[j], numbers[i]]; // 交换元素
            }
        }
        // 将 pivot 放到正确的位置
        [numbers[i + 1], numbers[end]] = [numbers[end], numbers[i + 1]];
        return i + 1; // 返回 pivot 的最终位置
    } 

    if (k <= 0 || k > numbers.length) {
        return [];
    }

    let len = numbers.length

    let start = 0
    let end = len - 1
    let index = partition(numbers,  start, end)
    // 找出第k大个数字，此时，k位置的左边都比k小，右边都比k大
    while(index !== k - 1) {
        // 如果下标大于k - 1，在左边部分的数组中
        if(index > k - 1) {
            end = index - 1
            index = partition(numbers, start, end)
        } else { // 如果下标小于k - 1，在右边部分的数组中
            start = index + 1
            index = partition(numbers, start, end)
        }
    }

    return numbers.slice(0, k);
}
// 思路4: 堆排序，构建最大堆，堆顶是最大的，然后依次清空就得到结果了


// 47、找出数组第k大的数字
// 思路：已经有固定套路o(n)时间复杂度的方法，基于快排到partition
//      如果基于数组的第k个数字调整，使得比第k个数字小的数组都位于数组左边，比第k个数字大的数字都位于数组右边，
//      这样位于k处的数字就是第k大的
function findK(numbers) {
    const partition = (numbers, start, end) => {
        let pivot = numbers[end]; // 选择最后一个元素作为基准（pivot）
        let i = start - 1; // i 用于跟踪小于或等于 pivot 的元素的最后一个索引
    
        for (let j = start; j < end; j++) { // 遍历从 start 到 end - 1 的元素
            if (numbers[j] <= pivot) { // 如果当前元素小于或等于 pivot
                i++; // 增加 i 的值
                [numbers[i], numbers[j]] = [numbers[j], numbers[i]]; // 交换元素
            }
        }
        // 将 pivot 放到正确的位置
        [numbers[i + 1], numbers[end]] = [numbers[end], numbers[i + 1]];
        return i + 1; // 返回 pivot 的最终位置
    } 

    if (k <= 0 || k > numbers.length) {
        return null;
    }

    let len = numbers.length

    let start = 0
    let end = len - 1
    let index = partition(numbers,  start, end)
    // 找出第k大个数字，此时，k位置的左边都比k小，右边都比k大
    while(index !== k - 1) {
        if(index > k - 1) {
            end = index - 1
            index = partition(numbers, start, end)
        } else { 
            start = index + 1
            index = partition(numbers, start, end)
        }
    }

    return numbers[k - 1]
}

// 48、数据流中的中位数, 难度：困难
// 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
// 如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值
// 分析：
//     数据结构选择分析：
//          数组：
//              优点：简单直接，对于插入和查找操作有一定的基础支持。
//              缺点：
//                  插入效率：如果数组没有排序，插入新数据时需要移动元素来保持数组有序，时间复杂度为 O (n)。
//                  查找中位数：在未排序的数组中查找中位数需要先排序，时间复杂度为 O (n)。
//          排序链表：
//              优点：
//                  插入操作：可以通过调整指针来快速找到插入位置，插入新数据的时间复杂度为 O (n)。
//                  查找中位数：通过两个指针可以在 O (1) 时间内找到链表的中间节点，从而计算中位数。
//              缺点：
//                  空间占用：需要额外的指针来维护链表结构，空间复杂度相对较高。
//          二叉搜索树：
//              优点：
//                  插入效率：可以通过递归的方式快速插入新数据，平均时间复杂度为 O (logn)。
//              缺点：
//                  平衡问题：如果二叉搜索树不平衡，例如退化成链表，插入和查找中位数的时间复杂度都会退化为 O (n)。
//                  查找中位数：需要在树中进行遍历和比较，时间复杂度难以保证在 O (1)。
//          AVL 树：
//              优点：
//                  平衡性能：通过调整树的结构来保持平衡，保证插入、删除和查找操作的时间复杂度都在 O (logn)。
//                  查找中位数：可以利用树的结构特点，在 O (1) 时间内得到中位数。
//              缺点：
//                  实现复杂：实现 AVL 树需要较高的编程技巧，在短时间内实现较为困难。
//          最大堆和最小堆：
//              优点：
//                  时间效率：插入新数据的时间复杂度为 O (logn)，可以快速得到堆顶元素，从而在 O (1) 时间内计算中位数。
//                  空间利用：只需要两个堆来维护数据，空间复杂度相对较低。
//              缺点：
//                  数据分布：需要保证数据在两个堆中的分布相对平衡，以确保中位数的计算准确。
// 最后选择最大堆和最小堆进行操作
// 思路：
//  所有数据进行从大到小排序
//  最大堆存储排序好的左边部分的数据，p1指针指向堆顶，堆顶是左边最大的值
//  最小堆存储排序好的右边部分的数据，p2指针指向堆顶，堆顶是右边最小的值
//  p1的值始终小于等于p2处的值
//  这样往堆中插入数据的时间复杂度是O (logn)，而可以用p1和p2直接找出中位数，时间复杂度是O (1)
class DynamicArray {
    constructor() {
        this.min = []; // 最小堆，堆顶元素最小，用于存储流数据的右半边数据
        this.max = []; // 最大堆，堆顶元素最大，用于存储流数据的左半边数据
    }

    insert(num) { // 这个方法解法不太懂啊，稍后研究研究
        // 总数为偶数时，将新数据插入最小堆
        if ((this.min.length + this.max.length) % 2 === 0) {
            if (this.min.length > 0 && num < this.min[0]) {
                // 如果新数字小于最小堆的根节点，将最小堆的根节点移到最大堆
                // 这里的思考解答： 
                //      如果新数字num小于最小堆的根节点，将最小堆的根节点移到最大堆，
                //      然后将小根堆的栈顶设为num，在进行两个堆调整的时候，大根堆里存在大于小根堆堆顶的元素，
                //      调整时又会重新放到小根堆，这样就实现了向小根堆插入节点的任务
                this.max.push(this.min[0]);
                this.min[0] = num; // 更新最小堆的根节点
                heapifyDown(this.min, 0); // 调整最小堆
            } else {
                // 否则直接插入最小堆
                this.min.push(num);
            }
            heapifyUp(this.min, this.min.length - 1); // 调整最小堆

            // 将最小堆的最小元素移动到最大堆
            if (this.min.length > this.max.length + 1) {
                const minTop = this.min.shift(); // 移除最小堆的根节点
                this.max.unshift(minTop); // 将其插入最大堆
                heapifyDown(this.max, 0); // 调整最大堆
            }
        } else { // 如果数据总数为奇数，将新数据插入最大堆
            if (this.max.length > 0 && num > this.max[0]) {
                // 如果新数字大于最大堆的根节点，将最大堆的根节点移到最小堆
                this.min.push(this.max[0]);
                this.max[0] = num; // 更新最大堆的根节点
                heapifyDown(this.max, 0); // 调整最大堆
            } else {
                // 否则直接插入最大堆
                this.max.push(num);
            }
            heapifyUp(this.max, this.max.length - 1); // 调整最大堆

            // 将最大堆的最大元素移动到最小堆
            if (this.max.length > this.min.length) {
                const maxTop = this.max.shift(); // 移除最大堆的根节点
                this.min.unshift(maxTop); // 将其插入最小堆
                heapifyDown(this.min, 0); // 调整最小堆
            }
        }
    }

    getMedian() {
        if (this.min.length === this.max.length) {
            // 总数为偶数时，计算中位数
            return (this.min[0] + this.max[0]) / 2;
        } else {
            // 总数为奇数时，返回最小堆的根节点
            return this.min[0];
        }
    }
}

// 堆调整函数：向下调整堆
function heapifyDown(arr, index) {
    const left = 2 * index + 1; // 左子节点索引
    const right = 2 * index + 2; // 右子节点索引
    let largest = index; // 初始化最大值为当前节点

    // 找到当前节点、左子节点和右子节点中的最大值
    if (left < arr.length && arr[left] < arr[largest]) {
        largest = left;
    }

    if (right < arr.length && arr[right] < arr[largest]) {
        largest = right;
    }

    // 如果最大值不是当前节点，则交换它们的值，并继续调整子树
    if (largest !== index) {
        [arr[index], arr[largest]] = [arr[largest], arr[index]];
        heapifyDown(arr, largest); // 递归调整
    }
}

// 堆调整函数：向上调整堆
function heapifyUp(arr, index) {
    while (index > 0) {
        const parent = Math.floor((index - 1) / 2); // 计算父节点索引
        // 如果当前节点小于其父节点，则交换它们的值，并继续向上调整
        if (arr[index] < arr[parent]) {
            [arr[index], arr[parent]] = [arr[parent], arr[index]];
            index = parent; // 更新当前索引为父节点索引
        } else {
            break; // 如果不需要交换，结束循环
        }
    }
}

// 49、连续子数组的最大和
// 输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。要求求出所有子数组的和的最大值，并且时间复杂度为o(n)。
// [1, -2, 3, 10, -4, 7, 2, -5]
// 思路：
//      创建一个变量记录sum，一个变量记录最大和res 从第一位开始累加
//      累加过程中若当前累加和<=0,则抛弃前面的累加和，从当前数字开始累加，累加和为当前数字值
//      累加过程中也一直比较更新最大累加和
function findGreatestSumOfSubArray(numbers) {
    if (numbers === null || numbers.length === 0) {
        return null;
    }
    let sum = 0
    let res = -Infinity
    for(let i = 0; i < numbers.length; i++) {
        if(sum <= 0) {
            sum = numbers[i]
        } else {
            sum += numbers[i]
        }

        if(sum > res) {
            res = sum
        }
    }
    return res
}

// 50、 判断一个数含1的个数
// 思路： 对求余数判断整数的个位数字是不是，如果大于，则除以之后再判断个位数字是不是，以此类推，计算每个数字中的个数并累加
// 时间复杂度是 o(logn)
function numberOf1(n) {
    if(!n) return 0

    let num = 0
    while(n) {
        if(n % 10 === 1) {
            num++
        }
        n = Math.floor(n / 10)
    }
    return num
}

// 51、1 - n 整数中 1 出现的次数， 难度：困难
// 输入一个整数，求这个整数的十进制表示中出现的次数。例如，输入12，1到12这些整数中包含的数字有1、10、11和12，1一共出现了5次。
// 思路1: 直接遍历每个数，对每个数求含1的个数,累加起来
// 思路2:   
//      以21345为例分析规律，将1-21345的数字分为两段：1-1345和1346-21345。
//       先看1346-21345中1出现的次数，分为1出现在最高位（万位）和出现在其他位的情况。
//      在最高位，1出现在10000-19999这个10000数字的万位中（对于万位是1的数字如12345，1出现的次数为除去最高数字之后剩下的数字再加1，即2346次）；
//      在其他位，1346-21345可再分成两段，每一段剩下的4位数字中，选择其中一位是1，其余三位可在中0-9任意选择，根据排列组合原则计算出1出现的次数。
//      对于1-1345中出现1的次数，可通过递归求得
// 函数 numberOf1Between1AndN 用于计算从 1 到 n 中数字 1 出现的次数
function numberOf1Between1AndN(n) {
    const numberOf1 = (strN) => {
        // 检查输入的字符串是否有效
        if (strN === null || strN[0] < '0' || strN[0] > '9' || strN === '') {
            return 0;
        }
        // 将字符串的第一个字符转换为数字
        let first = parseInt(strN[0]);
        // 获取字符串的长度
        let length = strN.length;
        // 如果长度为 1 且第一个数字为 0，则数字 1 的出现次数为 0
        if (length === 1 && first === 0) {
            return 0;
        }
        // 如果长度为 1 且第一个数字大于 0，则数字 1 的出现次数为 1
        if (length === 1 && first > 0) {
            return 1;
        }
        let numFirstDigit = 0;
        // 当第一个数字大于 1 时，计算该数字作为最高位时 1 出现的次数
        if (first > 1) {
            // 例如，对于三位数 2xx，最高位为 1 的情况有 100 到 199，共 10^(3-1) 种
            numFirstDigit = Math.pow(10, length - 1);
        } 
        // 当第一个数字等于 1 时，计算该数字作为最高位时 1 出现的次数
        else if (first === 1) {
            // 例如，对于 123，最高位为 1 的情况有 100 到 123，共 23 + 1 种
            numFirstDigit = parseInt(strN.slice(1)) + 1;
        }
        // 计算除去最高位外其他位上数字 1 出现的次数
        // first 表示最高位数字，(length - 1) 表示其他位的位数，Math.pow(10, length - 2) 表示每个位上出现 1 的可能组合
        let numOtherDigits = first * (length - 1) * Math.pow(10, length - 2);
        // 递归调用 numberOf1 计算除去最高位后的数字中 1 的出现次数，并将结果相加
        return numFirstDigit + numOtherDigits + numberOf1(strN.slice(1));
    }
    if (n <= 0) {
        return 0;
    }
    let strN = n.toString();
    return numberOf1(strN);
}

// 52、数字序列中某一位的数字
// 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等，
// 例如：输入n=3,输出3；输入n=11,输出0.求任意第n位对应的数字
// 思路： 
//      首先从最直观的方法开始思考，即从开始逐一枚举每个数字，计算每个数字的位数并累加。
//      当累加的数位大于时，说明第位数字在当前数字中。然后从该数字中找出对应的那一位。
//      为了提高效率，可以尝试寻找规律跳过若干数字。例如，先确定第n位在几位数中，再计算出具体是哪个数字的哪一位。
function findN(n) {
    if(!n) {
        return null
    }

    // 计算digits位数有多少个
    const countOfIntegers = (digits) => {
        if(digits === 1) {
            return 10
        }

        return 9 * Math.pow(10, digits - 1)
    }

    // 计算给定位数 digits 的数字范围的起始数字
    const  beginNumber = (digits) => {
        // 当位数为 1 时，起始数字为 0
        if (digits === 1) {
            return 0;
        }
        // 当位数大于 1 时，计算起始数字，例如 2 位数从 10 开始
        return Math.pow(10, digits - 1);
    }

    const findCore = (n, digits) => {
        // 计算数字所在的数字范围的起始数字
        let number = beginNumber(digits) + Math.floor(n / digits);
        // 计算从右往左数的索引
        let indexFromRight = digits - n % digits;
        // 从右往左移动数字，直到找到所需的数字
        for (let i = 1; i < indexFromRight; i++) {
            number = Math.floor(number / 10);
        }
        // 返回该数字的最后一位
        return number % 10;
    }

    // 初始化数字的位数为 1
    let digits = 1;
    while(true) {
        // 计算该位数的个数
        let numbers = countOfIntegers(digits)

        // 如果索引在当前位数的数字范围内，就去这个范围内找
        if(n < numbers * digits) {
            return findCore(n, digits)
        } else {
            // 否则，跳过digits * numbers位数字，目标数字在紧跟着后面的第n - digits * numbers位
            n = n - digits * numbers
            // 增加位数继续查找
            digits++
        }
    }
}

// 53、把数组排成最小的数
// 输入一个正整数数组，将数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
// 例如[3, 32, 321]，输入数组，则打印出这3个数字能排成的最小数字321323
// 思路： 需要对数组元素进行排序，排序规则，假设一个是m，一个是n，如果mn < nm, 则m排在n前面
// 这里隐藏了一个问题就是，有可能结果超出范围，所以结果和比较大时候用字符串形式表示
function printMinNumber(numbers) {
    const compare = (m, n) => {
        const mn = '' + m + n
        const nm = '' + n + m
        if(mn < nm) {
            return -1
        } else if(mn > nm) {
            return 1
        } else {
            return 0
        }
    }
    

    if(!numbers || numbers.length === 0) {
        return null
    }
   
    numbers.sort(compare)

    let res = ''
    for(let i = 0; i < numbers.length; i++) {
        res += numbers[i]
    }
    return res
}

// 54、把数字翻译成字符串
// 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
// 一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法
// 思路：动态规划
//      以12258为例，分析从数字的第一位开始计算不同翻译方法数目的过程。
//      对于第一位数字1，有两种选择：单独翻译成 “b”，后面剩下数字2258；或者1和紧挨着2的一起翻译成 “m”，后面剩下数字258。
//      当最开始的一个或两个数字被翻译成一个字符后，接着翻译后面剩下的数字
//      采用从数字末尾开始，从右到左翻译并计算不同翻译数目，可以避免重复子问题的出现
function getTranslationCount(num) {
    const countCore = (str) => {
        let len = str.length
        // 创建一个数组，存放中间态结果，
        // 如counts[len - 1]就是最后一位开始的翻译种类个数，
        // counts[len - 2]就是倒数第二数字开始的翻译种类个数，
        // 以此类推，counts[0]就是从第一位开始的整个数字的翻译种类个数
        let counts = new Array(len).fill(0)
        let count = 0
        // 从右到左翻译可以避免出现重复子问题
        for(let i = len - 1; i >= 0; i--) {
            count = 0
            if(i < len - 1) {
                // 如果不是最后一位，取后一位的结果
                count = counts[i + 1]

                // 取当前位和后一位的值
                let digit1 = parseInt(str[i])
                let digit2 = parseInt(str[i + 1])
                // 两位组成一个数值
                let converted = digit1 * 10 + digit2
                // 如果值在10到25范围内，则可以合并翻译成l - z之间的字母
                if(converted >= 10 && converted <= 25) {
                    // 如果i是倒数第二位，则后两位合并后只能位总种类数+1
                    if(i === len - 2) { 
                        count += 1
                    } else { // 否则第i位和第i+1位合并，总种类数会多出counts[i + 2]个
                        count += counts[i + 2]
                    }
                }
            } else { 
                // 最后一位开始的翻译种类只会有1个
                count = 1
            }

            // 为当前状态赋值
            counts[i] = count
        }

        return counts[0]
    }

    if(num < 0) {
        return 0
    }

    num = num.toString()

    return countCore(num)
}

// 55、礼物的最大价值
// 在一个m*n的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于0）。
// 可以从棋盘的左上角开始拿格子里的礼物，并每次向左或者向下移动一格，直到到达棋盘的右下角。要求计算最多能拿到多少价值的礼物。
// 例如，在下面的棋盘中，如果沿着带下划线的数字的线路（1、12、5、7、7、16、5），能拿到最大价值为53的礼物
// 1  10 3  8
// 12 2  9  6
// 5  7  4  11
// 3  7  16 5
// 思路：动态规划，使用一个二维数组保存走到每个节点时价值的最大值，从(0,0)开始走，走到右下角时数组里的值最大
function getMaxValue(values, rows, cols) {
    if(!values || rows === 0 || cols === 0) {
        return 0
    }

    let maxValues = new Array(rows).fill(0).map(() => new Array(cols).fill(0))
    // 从(0,0)开始走
    for(let i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            // 移动可以有两个方向，up或left
            let up = 0
            let left = 0
            if(i > 0) { // 当前行大于0则可以向上走
                up = maxValues[i - 1][j]
            }

            if(j > 0) { // 当前列大于0则可以向左走
                left = maxValues[i][j - 1]
            }

            // 当前格子所在的最大值就是可以移动到当前格子的最大值加上当前格子的值
            if(up > left) {
                maxValues[i][j] = up + values[i * cols + j]
            } else {
                maxValues[i][j] = left + values[i * cols + j]
            }
        }
    }
    return maxValues[rows - 1][cols -1]
}

// 56、最长不含重复字符的子字符串
// 要求从给定字符串中找出一个最长的不包含重复字符的子字符串，并计算其长度。假设字符串中只包含a-z的字符。
// 例如，在字符串 “arabcacfr” 中，最长的不含重复字符的子字符串是 “acfr”，长度为4
// 思路：动态规划，用一个map来存储字符上一次的位置，用start记录字符串的其实位置，如果当前字符已经有上一次的位置了，则证明已经重复了，就更新起始
// 
function longestSubstringWithoutDuplication(str) {
    // 存储字符在字符串中的位置
    let dic = new Map()
    let res = 0
    let start = -1; // 用于记录当前子字符串的起始位置
    for (let i = 0; i < str.length; i++) {
        // 获取当前字符上一次出现的位置
        let pos = dic.has(str[i]) ? dic.get(str[i]) : -1; // 如果字符存在于 Map 中，获取其位置；否则为 -1

        // 更新子字符串的起始位置
        if (pos !== -1) { // 每次pos值不为-1就可以直接更新start为pos?因为遍历是从0开始的，只要pos有值，它肯定比当前的start大
            start = pos;
        }

        dic.set(str[i], i); // 更新当前字符的位置

        // 计算当前子字符串的长度
        let tmp = i - start; // 当前子字符串的长度为当前索引 i 减去起始位置
        res = Math.max(res, tmp); // 更新结果，取当前最长长度和之前记录的最长长度的较大值
    }
    return res;
}

// 57、丑数
// 把只包含因子2、3和5的数称作丑数（Ugly Number），习惯上把1当作第一个丑数。要求按从小到大的顺序求出第1500个丑数。例如，6、8都是丑数，但14不是，因为它包含因子7。
// 思路1: 逐个判断数是不是丑数
const isUgly = (number) => {
    while(number % 2 === 0) {
        number = number / 2
    }

    while(number % 3 === 0) {
        number = number / 3
    }

    while(number % 5 === 0) {
        number = number / 5
    }

    return number === 1
}

function getUglyNumber(index) {
    if(index <= 0) {
        return null
    }

    let curIndex = 0
    let number = 0
    while(curIndex < index) {
        number++
        if(isUgly(number)) {
            curIndex++
        }
    }
    return number
}
// 思路2: 使用一个数组来存储已经排序好的丑数，第一个丑数是1，从第二个丑数开始，每个丑数的值都是之前的丑数乘以2、3、5得出的值
//  只要构建从小到大排序好的丑数数组，直到数组长度为n时，最后一个数就是结果
function getUglyNumber(index) {
    if(index <= 0) {
        return null
    }

    let numbers = [1]
    // 分别是指向当前需要乘以 2、3、5 的丑数在 numbers 数组中的位置。
    let p2 = 0
    let p3 = 0
    let p5 = 0

    while(numbers.length < index) {
        let res = Math.min(numbers[p2] * 2, numbers[p3] * 3, numbers[p5] * 5)
        numbers.push(res)

        // 避免重复，确保每个丑数只使用一次
        // 移动指针指针指向下一个可以乘2｜3｜5的数
        if(numbers[p2] * 2 === res) {
            p2++
        }

        if(numbers[p3] * 3 === res) {
            p3++
        }

        if(numbers[p5] * 5 === res) {
            p5++
        }
    }

    return numbers[index - 1]
}

// 这个方法其实可以抽成一个通用的套路，factors数组是满足条件的因数数组，这样就可以输出任意第index个满足只有factors中因数的数了
function getSpecialNumber(index, factors) {
    if (index <= 0 || factors.length === 0) {
        return null;
    }

    let numbers = [1]; // 初始数组，第一个特殊数是 1
    let pointers = Array(factors.length).fill(0); // 每个因数的指针初始化为 0

    while (numbers.length < index) {
        // 计算所有因数对应的候选值
        let candidates = factors.map((factor, i) => numbers[pointers[i]] * factor);

        // 选择当前的最小值作为下一个特殊数
        let nextNumber = Math.min(...candidates);
        numbers.push(nextNumber);

        // 更新指针
        for (let i = 0; i < factors.length; i++) {
            if (candidates[i] === nextNumber) {
                pointers[i]++;
            }
        }
    }

    return numbers[index - 1];
}

// 58、第一个只出现一次的字符
// 在字符串中找出第一个只出现一次的字符。例如，输入 “abaccdef”，输出 “b”
// 思路：哈希表存储次数，找到大于1的就返回，时间复杂度是o(n), 空间复杂度是字符的种类数o(k)
function firstNotRepeatingChar(str) {
    if(!str || str.length === 0) {
        return null
    }
    let map = new Map()
     // 第一次遍历：统计每个字符的出现次数
     for (let char of str) {
        map.set(char, (map.get(char) || 0) + 1);
    }

    // 第二次遍历：找到第一个出现次数为 1 的字符
    for (let char of str) {
        if (map.get(char) === 1) {
            return char;
        }
    }

    return null;
}

// 59、找出字符流中第一个只出现一次的字符。
// 例如，当从字符流中只读出前两个字符 “go” 时，第一个只出现一次的字符是 “g”；
// 当从该字符流中读出前6个字符 “google” 时，第一个只出现一次的字符是 “l”。字符只能一个接着一个从字符流中读出来
// 思路：定义一个数据容器来保存字符在字符流中的位置。
//      当字符第一次出现时，记录其位置；当再次出现时，将其位置更新为特殊值（如负数值）。
//      这里用哈希表。通过扫描整个map，找出最小的大于等于0的值对应的字符，即为第一个只出现一次的字符
class CharStatistics {
    constructor() {
        this.index = 0;
        this.map = new Map();
    }

    insert(ch) {
        // 如果字符还未出现过，记录其索引
        if (!this.map.has(ch)) {
            this.map.set(ch, this.index);
        }
        // 如果字符已经出现过且未标记为重复，标记为重复
        else if (this.map.get(ch) >= 0) {
            this.map.set(ch, -2);
        }
        this.index++;
    }

    firstAppearingOnce() {
        for (let [char, index] of this.occurrence) {
            if (index >= 0) {
                return char; // 找到并返回第一个未重复的字符
            }
        }
        return '';
    }
}

// 60、数组中的逆序对， 难度：困难， 先不做了
// 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，要求求出这个数组中的逆序对的总数。
// 例如，在数组[7,5,6,4]中，一共存在5个逆序对，分别是(7,5)、(7,6)、(7,4)、(5,4)和(6,4)。
function inversePairs(numbers) {}

// 61、两个链表的第一个公共节点
// 输入两个链表，找出他们的第一个公共节点
// 思路1： 链表1长度m，链表2长度n，链表1每遍历一个节点，就遍历一遍链表2找有没有相同的，时间复杂度是o(mn),显然不可取
// 思路2: 分别把两个链表打进两个栈里，然后两个栈一起出栈，在公共节点出现之前，出栈的值都会相同，所以一旦出栈的值开始不同，证明前一个出栈的值是公共节点，
//      时间复杂度o(m+n),空间复杂度o(m+n)
function findFirstCommonNode(l1, l2) {
    if (!l1 || !l2) {
        return null;
    }

    let stack1 = [];
    let stack2 = [];

    // 第一个链表入栈
    while (l1) {
        stack1.push(l1);
        l1 = l1.next;
    }

    // 第二个链表入栈
    while (l2) {
        stack2.push(l2);
        l2 = l2.next;
    }

    let res = null;
    while (stack1.length && stack2.length) {
        let a = stack1.pop();
        let b = stack2.pop();
        if (a === b) {
            res = a; // 返回公共节点本身
        } else {
            break; // 遇到不相等的节点，停止比较
        }
    }
    return res;
}
// 思路3: 跟链表中的第k个节点的解法类似，先统计l1和l2的长度，l1如果比l2长k，那么l1先走k步，接着两个链表同时遍历，找到第一个相同的节点就是目标节点
//      时间复杂度是o(m+n),空间复杂度是o(1)
function findFirstCommonNode(l1, l2) {
    if (!l1 || !l2) {
        return null;
    }

    const getLen = (l) => {
        let len = 0;
        while (l) {
            len++;
            l = l.next;
        }
        return len;
    }

    let len1 = getLen(l1);
    let len2 = getLen(l2);
    let gap = Math.abs(len1 - len2);

    // 调整较长链表的指针，使两个链表从相同的起始位置开始同步遍历
    while (gap) {
        if (len1 > len2) {
            l1 = l1.next;
        } else {
            l2 = l2.next;
        }
        gap--;
    }

    // 同步遍历两个链表，寻找第一个公共节点
    while (l1 && l2) {
        if (l1 === l2) {
            return l1;
        }
        l1 = l1.next;
        l2 = l2.next;
    }

    // 没有公共节点
    return null;
}

// 62、在排序数组中查找数字
// 统计一个数字在排序数组中出现的次数。例如，输入排序数组[1,2,3,3,3,3,4,5]和数字3，由于在这个数组中3出现了4次，所以输出4
// 思路： 典型的二分查找,根据排序数组的特性，先找第一个出现的位置，再找第二个出现的位置，两个位置都知道了，自然知道结果了
// 二分查找很适合在排序数组里使用
function getNumberOfK(numbers, k) {
    if(!numbers || numbers.length === 0) {
        return 0
    }

    const firstK = (numbers, k, left, right) => {
        if(left > right) {
            return -1
        }

        let mid = (left + right) >> 1
        if(numbers[mid] === k) { // 找到了
            // 找第一个k，边界条件应该是它的前一个值不等于k的值，这时候这个k才是第一个
            if(mid > 0 && numbers[mid - 1] === k) {
                return firstK(numbers, k, left, mid - 1)
            } else {
                return mid
            }

        } else if(numbers[mid] > k) { // 大于，再mid左边找
            return firstK(numbers, k, left, mid - 1)
        } else { // 小于在mid右边找
            return firstK(numbers, k, mid + 1, right)
        }
    }

    const lastK = (numbers, k, left, right) => {
        if(left > right) {
            return -1
        }

        let mid = (left + right) >> 1
        if(numbers[mid] === k) { // 找到了
            // 找最后一个k，边界条件应该是它的后一个值不等于k的值，这时候这个k才是最后一个k
            if(mid < right && numbers[mid + 1] === k) {
                return lastK(numbers, k, mid + 1, right)
            } else {
                return mid
            }

        } else if(numbers[mid] > k) { // 大于，再mid左边找
            return lastK(numbers, k, left, mid - 1)
        } else { // 小于在mid右边找
            return lastK(numbers, k, mid + 1, right)
        }
    }

    // 第一次出现的位置
    let first = firstK(numbers, k, 0, numbers.length - 1)
    // 最后出现的位置
    let last = lastK(numbers, k, 0, numbers.length - 1)

    if(first !== -1 && last !== -1) {
        return last - first + 1
    }

    return 0
}

// 63、0 - n中缺失的数字
// 一个长度为n - 1度递增排序数组中所有数字都是唯一的， 并且每个数字都在0 - n-1之内，在范围0 - n-1的n个数字有且只有一个数字不在该数字中，找出该数字
// 思路： 二分查找，找出第一个下标和值不相等的数
function findNumber(numbers) {
    if(!numbers || numbers.length === 0) {
        return -1
    }

    const findCore = (numbers, left, right) => {
        if(left > right) {
            return -1
        }
        let mid = (left + right) >> 1
        if(mid !== numbers[mid]) { // 如果下标和值不相等
            // 如果 mid 是第一个位置，或者前一个位置值与下标相等，说明找到了
            if(mid === 0 || numbers[mid - 1] === mid - 1) {
                return mid
            } else { // 否则证明在左边
                return findCore(numbers, left, mid - 1)
            }
        } else { // 如果下标和值相等，证明在右边
            return findCore(numbers, mid + 1, right)
        }
    }

    return findCore(numbers, 0, numbers.length - 1)
}

// 64、数组中数值和下标相等的元素
// 假设一个单调递增的数组里每个元素都是整数并且是唯一的。请实现一个函数，找出数组中任意一个数值等于其下标的元素
// 思路：还是二分查找，找出第一个值和下标相等的
function findEqual(numbers) {
    if(!numbers || numbers.length === 0) {
        return -1
    }

    const findCore = (numbers, left, right) => {
        if(left > right) {
            return -1
        }

        let mid = (left + right) >> 1

        if(mid === numbers[mid]) {
            return mid
        } else if(mid > numbers[mid]) { // 在左边
            return findCore(numbers, left, mid - 1)
        } else { // 在右边
            return findCore(numbers, mid + 1, right)
        }
    }

    return findCore(numbers, 0, numbers.length - 1)
}


// 65、二叉搜索树的第 k 大节点
// 给定一棵二叉搜索树，请找出其中第k大的节点。
//       5
//      / \
//     3   7
//    /\   /\
//   2  4 6  8
// 例如，在上面的二叉搜索树中，按节点数值大小顺序，第三大节点的值是4
// 思路： 中序遍历是递增的，所以中序遍历即可, 但是正常的中序遍历是从到小，而要从大小，一个是右子树->根->左子树
function kthLargest(root, k) {
    if(!root || k === 0) {
        return null
    }
    let res
    let count = 0
    const inOrder = (node) => {
        if(node === null || count >= k) {
            return 
        }

        inOrder(node.right)
        count++
        if(count === k) {
            res = node.val
            return
        }
        inOrder(node.left)
    }
    inOrder(root)
    return res
}

// 66、二叉树的深度
// 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
//       1
//      / \
//     2   3
//    /\    \
//   4  5    6
//     /
//    7
// 例如，在上面二叉树中，最长路径为1->2->5->7，所以深度为4
// 思路： 如果只有一个根节点深度是1，如果有左右子树，那么深度就是左右子树深度的最大值 + 1
function treeDepth(root) {
    if(!root) {
        return 0
    }

    let leftDepth = treeDepth(root.left)
    let rightDepth = treeDepth(root.right)

    return Math.max(leftDepth, rightDepth) + 1
}


// 67、平衡二叉树
// 输入一颗二叉树的根节点，判断是不是平衡二叉树，如果某二叉树的任意节点的左右子树深度相差不超过1，那么它就是一棵平衡二叉树
// 上题的例子就是一颗平衡二叉树
// 思路1: 基于二叉树深度来求平衡二叉树
//       次调用treeDepth都需要重新计算子树的深度，导致重复计算。对于深度为 n 的树，复杂度为o(n2)
function isBalanceTree(root) {
    if(!root) {
        return true
    }

    let leftDepth = treeDepth(root.left)
    let rightDepth = treeDepth(root.right)

    if(Math.abs(leftDepth - rightDepth) > 1) {
        return false
    }

    return isBalanceTree(root.left) && isBalanceTree(root.right)
}
// 思路2: 在判断平衡的同时，计算了每个节点的深度，避免重复递归
function isBalanceTree(root, depth = 0) {
    if(!root) {
        depth = 0
        return true
    }
    // 递归检查左子树和右子树是否平衡，同时计算深度
    const checkBalance = (node) => {
        if (!node) {
            return 0;  // 空节点的深度为0
        }

        let left = checkBalance(node.left);  // 左子树深度
        if(left === -1) return -1

        let right = checkBalance(node.right);  // 右子树深度
        if(right === -1) return -1

        // 如果左右子树的深度差大于1，则说明不平衡
        if (Math.abs(left - right) > 1) {
            return -1;  // 返回-1表示不平衡
        }

        // 返回当前节点的深度
        return Math.max(left, right) + 1;
    }

    return checkBalance(root) !== -1;  // 如果返回-1，表示不平衡，否则表示平衡
}

// 68、数组中只有一个数出现了一次其他都出现了两次，找出哪个数字
// 思路： 任何一个数异或自己得出的值都是0，也就是8^8 = 0,把所有数字连续异或，相同的数就会消掉，只剩下只有一次的那个数
function findAppearOne(numbers) {
    if(!numbers || numbers.length === 0) {
        return null
    }

    let res = numbers[0]
    for(let i = 1; i < numbers.length; i++) {
        res = res ^ numbers[i]
    }
    return res
}

// 69、数组中数字出现的次数
// 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是o(n)，空间复杂度是o(1)。
// 例如，在数组[2,4,3,6,3,2,5,5]中，只出现一次的数字是4和6
// 思路：如果一个数组里只有一个出现1次的数，所有数异或就能得出结果，现在数组里有两个出现一次的数，所以想办法把数组拆分成两个都只包含一个出现一次的数组就可以了
//      a、先异或所有数得到的是两个只出现一次数字的异或结果，然后因为这两个数字不同，所以异或结果不为0，
//      b、在异或结果中找到第一个为1的位，根据这一位将数组分为两组，一组该位为1，另一组该位为0。
//      c、两个只出现一次的数字分别在这两组中，且相同的数字会被分到同一组（因为相同的数字任意一位都是相同的，所以不可能把两个相同的数字分到不同的组里去）。
//      d、然后分别对两组数字进行异或，就能得到这两个只出现一次的数字
function findNumsAppearOnce(numbers) {
    if(!numbers || numbers.length === 0) {
        return null
    }

    // 找出二进制中第一个1出现在第几位
    const findFirstBit1 = (num) => {
        let indexBit = 0
        // num & 1检测最低位是不是1
        while((num & 1) === 0) {
            num = num >> 1 // 右移1为，说明1不在当前位
            indexBit++ // 位数+1
        }
        return indexBit
    }

    // 判断二进制数第bitIndex是是不是1
    const isBit1 = (num, bitIndex) => {
        num = num >> bitIndex
        return (num & 1) === 1
    }

    // 1、求出两个出现一次数字的异或结果
    let resOR = numbers[0]
    for(let i = 1; i < numbers.length; i++) {
        resOR = resOR ^ numbers[i]
    }

    // 2、找出结果的二进制第一位1出现在第几位,依靠它进行分组
    let firstBit = findFirstBit1(resOR)

    // 定义两个结果
    let num1 = 0
    let num2 = 0
    for(let i = 0; i < numbers.length; i++) {
        // 在循环过程中根据firstBit对结果进行分组
        if(isBit1(numbers[i], firstBit)) {
            num1 = num1 ^ numbers[i]
        } else {
            num2 = num2 ^ numbers[i]
        }
    }
    return [num1, num2]
}

// 70、在一个数组中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
//  例如，在数组中[1,1,1,2,2,2,3]，只出现一次的数字是3
// 思路：考虑数字的二进制表示，对于出现三次的数字，其每一位上出现1的次数必然是的3倍数（0次、3次等）。
//      统计数组中所有数字每一位上出现1的次数，然后将每一位上1出现的次数对3取余，结果为1的位就是只出现一次的数字在该位上的值
function findNumsAppearOnce(numbers) {
    if(!numbers || numbers.length === 0) {
        return null
    }

    // 用来统计数组中所有数的每位二进制位上出现1的总次数
    let bitSum = new Array(32).fill(0)

    // 进行统计
    // 循环每个数据
    for(let i = 0; i < numbers.length; i++) {
        let bitMask = 1 //初始化1，先去判断最低位是不是1
        // 统计该数中每位的1的个数
        for(let j = 31; j >= 0; j--) {
            if(numbers[j] & bitMask) { // 判断当前位是不是1
                bitSum[j]++
            }
            bitMask = bitMask << 1 //左移一位，为下一次判断再前一位做准备
        }
    }

    let res = 0
    // 对刚刚统计的数据进行遍历
    for(let i = 0; i < 32; i++) {
        // 如果当前位的1个数不能被3整除，证明当前位的多余的1是这个只出现一次的数贡献的
        if(bitSum[i] % 3 !== 0) {
            res = res + Math.pow(2, i)
        } 
    }

    return res
}

// 71、和为 s 的数字
// 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于，则输出任意一对即可。
// 例如，输入数组[1,2,4,7,11,15]和数字15，由于4+11 = 15，所以输出4和11。
// 思路：双指针，一个在首位，一个在末尾
function findNumbersWithSum(numbers, s) {
    if(!numbers || numbers.length === 0 || s === undefined) {
        null
    }
    let p1 = 0
    let p2 = numbers.length - 1
    while(p1 < p2) {
        if(numbers[p1] + numbers[p2] > s) {
            p2--
        } else if(numbers[p1] + numbers[p2] < s) {
            p1++
        } else {
            return [numbers[p1], numbers[p2]]
        }
    }
    return []
}

// 72、输入一个正数s，打印出所有和为的连续正数序列（至少含有两个数）。
// 例如，输入15，由于1+2+3+4+5=4+5+6=7+8=15，所以打印出3个连续序列1-5、4-6和7-8。
// 思路： 双指针，一个指向首位，一个指向第二位，然后进行遍历
function findListWithSum(s) {
    if(s < 3) {
       return []
    }

    // p1 指向较小的数，p2指向较大的数
    let p1 = 1
    let p2 = 2
    let mid = (1 + s) >> 1 // p1一旦指向大于mid时就不可能找到值了
    let sum = p1 + p2
    let res = []
    while(p1 < mid) {
        if(sum === s) {
            let arr = []
            for(let i = p1; i <= p2; i++) {
                arr.push(i)
            }
            res.push(arr)
            // 找到满足条件的序列后，增大 big 以继续寻找下一个可能的序列
            p2++;
            sum += p2;
        } else if(sum > s) {
            sum -=  p1
            p1++
        } else {
            p2++
            sum += p2
        }
    }
    return res
}

// 73、翻转单词顺序
// 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。
// 例如，输入字符串 "I am a student."，则输出 "student. a am I"
// 思路： 首先反转整个句子，然后再反转每个单词,也可以一边反转句子一边反转单词
function reverseWords(str) {
    str = str.trim();
    let word = ''; // 存储当前单词
    let resultArr = [];

    // 遍历字符串，从右到左
    for (let i = str.length - 1; i >= 0; i--) {
        if (str[i] !== ' ') {
            word = str[i] + word; // 拼接当前单词
        } else if (word) {
            resultArr.push(word); // 遇到空格，保存当前单词
            word = '';
        }
    }

    // 处理最后一个单词
    if (word) {
        resultArr.push(word);
    }

    return resultArr.join(" ");
}

// 74、字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。
// 比如，输入字符串 "abcdefg" 和数字，该函数将返回左旋转位得到的结果 "cdefgab"
// 思路： 可以先将字符串的前个字符反转，再将剩余的字符反转，最后将整个字符串反转。
//      例如，对于 "abcdefg" 和，先反转 "ab" 得到 "ba"，再反转 "cdefg" 得到 "gfedc"，最后反转 "bagfedc" 得到 "cdefgab"
function leftRotateString(str, n) {
    if (!str || n === 0) {
        return str;
    }

    const reverse = (str) => {
        let res = ''
        for(let i = str.length - 1; i >= 0; i--) {
            res += str[i]
        }
        return res
    }

    // 防止 n 大于字符串长度
    n = n % str.length

    // 第一步、先反转前n个字符
    let first = reverse(str.slice(0, n))
    let second = reverse(str.slice(n))
    let res = first + second

    return reverse(res)
}
// 也可以直接用slice
function leftRotateString(str, n) {
    if (!str || n === 0) {
        return str;
    }

    // 防止 n 大于字符串长度
    n = n % str.length;

    // 拼接字符串：将从 n 到末尾的部分，移到前面
    return str.slice(n) + str.slice(0, n);
}

// 75、队列最大值
// 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
// 例如，输入 nums = [1,3,-1,-3,5,3,6,7]，k = 3，则滑动窗口依次为 [1,3,-1]、[3,-1,-3]、[-1,-3,5]、[-3,5,3]、[5,3,6]、[3,6,7]，
// 对应的最大值依次为 3、3、5、5、6、7。
// 思路1: 暴力扫描每个滑动窗口的最大值，时间复杂度是o(nk),k为滑动窗口的大小
// 思路2: 滑动窗口其实可以当作一个队列，向右移动的过程中相当于右边新纳入窗口的元素入队，左边离开窗口的元素出队
//      那么只要在入队出队过程中找出最大值就好了，如果可以用两个栈实现这个队列，就可以实现在o(1)时间内得到栈中的最大值，总体时间的时间复杂度就降到了o(n)
class MaxQueue {
    constructor() {
        this.quene = [] 
        this.maxQueue = []
    }

    addTail(ele) {
        this.quene.push(ele)
        // 弹出maxQuene栈顶比当前入队的值的小的值
        while(this.maxQueue.length && this.maxQueue[this.maxQueue.length - 1] < ele) {
            this.maxQueue.pop()
        }
        // 再入队，确保maxQueue是从大到小排列的
        this.maxQueue.push(ele)
    }

    deleteHead() {
       if(this.quene.length === 0) {
            return -1
       }
       const tobeDeleted = this.quene.shift()
       // 调整maxQueue
       if(tobeDeleted === this.maxQueue[0]) {
            this.maxQueue.shift()
       }
    }

    maxValue() {
        // 第一个值永远是最大的
        return this.maxQueue.length? this.maxQueue[0] : -1;
    }
}
function maxValues(numbers, k) {
    if (!numbers || numbers.length === 0 || k <= 0 || k > numbers.length) {
        return []
    }

    let queue = new MaxQueue()  // 将 maxQuene 更正为 MaxQueue
    // 先入队k个元素，实现第一个滑动窗口
    for (let i = 0; i < k; i++) {
        queue.addTail(numbers[i])
    }

    let res = []
    res.push(queue.maxValue())  // 获取第一个滑动窗口的最大值

    // 然后让窗口滑动
    for (let i = k; i < numbers.length; i++) {
        queue.addTail(numbers[i])  // 将新的元素加入队列
        queue.deleteHead()         // 删除队列头部元素
        res.push(queue.maxValue()) // 获取当前滑动窗口的最大值
    }

    return res
}

// 76、n个骰子的点数
// 把 n 个骰子扔在地上，所有骰子朝上一面的点数之和为 s。输入 n，打印出 s 的所有可能的值出现的概率
// 总体思路： n个骰子最小的和为n，最大的和为6n，排列组合数是6的n次方，统计出每个点数出现的次数，再除以6的n次方，就是每个点数和的概率
// 思路1: 动态规划， 定义一个递归函数来计算个骰子点数之和为s的组合数。对于每个骰子，它可能出现1到6点，
//      所以可以通过递归计算前n-1个骰子在不同点数情况下与当前骰子点数组合得到的s组合数。
//      为了避免重复计算，使用一个二维数组（记忆化数组）来存储已经计算过的结果
function printP(n) {
    if (n < 1) {
        return []
    }

    // 求总共的组合数
    const getCount = (n, s) => {
        // 如果总和s小于n或大于6n，说明不可能出现这种组合
        if (s < n || s > 6 * n) return 0;
        if (n === 1) {
            // 只有1个骰子时，点数和为1到6的概率均为1
            return s >= 1 && s <= 6 ? 1 : 0;
        }
        if (memo[n][s] !== -1) return memo[n][s];
        
        let sum = 0;
        // 递归地计算前n-1个骰子的组合数
        for (let i = 1; i <= 6; i++) {
            sum += getCount(n - 1, s - i);
        }
        memo[n][s] = sum;
        return sum;
    }

    // 定义二维数组统计计算结果
    // memo[i][j]表示当有i个骰子时，点数之和为j的组合数
    const memo = new Array(n + 1).fill(0).map(() => new Array(6 * n + 1).fill(-1))
    const total = Math.pow(6, n)
    let res = []

    // 逐个求点数和从n到6n的概率值
    for (let i = n; i <= 6 * n; i++) {
        res.push(getCount(n, i) / total)
    }

    return res
}
// 思路2: 动态规划，上面是自顶向下的求解，现在实现自底向上的版本
function printP(n) {
    if (n < 1) {
        return []
    }

    // 定义二维数组统计计算结果
    // memo[i][j]表示当有i个骰子时，点数之和为j的组合数
    const memo = new Array(n + 1).fill(0).map(() => new Array(6 * n + 1).fill(0))

    // 初始化1个骰子的情况
    for(let i = 1; i <=6; i++) {
        memo[1][i] = 1
    }

    // 控制骰子的数量,逐步考虑从 2 个骰子到 n 个骰子的情况。
    for(let i = 2; i <= n; i++) {
        // 控制骰子点数的总和，遍历了 i 个骰子所有可能的点数总和范围
        for(let j = i; j <= 6 * i; j++) {
            // 表示当前骰子可能掷出的点数，范围是从 1 到 6
            for(let k = 1; k <= 6 && k <= j; k++) {
               // 当我们考虑第 i 个骰子掷出 k 点时，要得到 i 个骰子点数总和为 j，就需要在前 i - 1 个骰子掷出 j - k 点的基础上
               // 由于第 i 个骰子可以掷出 1 到 6 中的任何一个点数（由 k 表示），
               // 所以将 memo[i - 1][j - k] 的值累加到 memo[i][j] 中，将所有可能的 k 值对应的组合数相加，就能得到 i 个骰子点数总和为 j 的组合数。
               memo[i][j] += memo[i - 1][j - k] // i - 1 个骰子掷出点数总和为 j - k 的组合数。
            }
        }
    }

    const total = Math.pow(6, n)
    let res = []
     // 逐个求点数和从n到6n的概率值
     for (let i = n; i <= 6 * n; i++) {
        res.push(memo[n][i] / total)
    }

    return res
}

// 77、扑克牌中的顺子
// 从扑克牌中随机抽 5 张牌，判断是不是一个顺子，这里的顺子指的是连续的 5 张牌。
// 2～10 为数字本身，J 为 11，Q 为 12，K 为 13，A 为 1，而大小王可以看成任意数字。
// 思路1: 因为会输入大小王，大小王的值为0，大小王可以作为任何数值，所以需要统计0的数量，
//      然后统计所有数直接的总间隔，如果间隔数不超过大小王也就是0的数量，那这些间隔中缺失的数字就可以用大小王表示，组成顺子
function isStraight(nums) {
    // 对数组进行排序，方便后续处理
    nums.sort((a, b) => a - b);
    let numberOfZero = 0;
    for (let i = 0; i < nums.length - 1; i++) {
        if (nums[i] === 0) {
            // 统计大小王的数量
            numberOfZero++;
        } else if (nums[i] === nums[i + 1]) {
            // 检查是否有重复元素，有重复元素则不是顺子
            return false;
        }
    }
    let numberOfGap = 0;
    for (let i = 0; i < nums.length - 1; i++) {
        if (nums[i]!== 0 && nums[i + 1]!== 0) {
            // 计算相邻非零元素的间隔
            numberOfGap += nums[i + 1] - nums[i] - 1;
        }
    }
    // 比较间隔数量和大小王数量，如果间隔能被大小王填补则是顺子
    return numberOfGap <= numberOfZero; 
}
// 思路2:用一个集合来存储非大小王的牌，同时找出最大和最小的非大小王牌。
//      若集合的大小加上大小王的数量等于 5，且最大非大小王牌与最小非大小王牌的差小于 5，则是顺子。
function isStraight(nums) {
    let set = new Set()
    let min = Infinity
    let max = -Infinity
    let zeroCount = 0;
    for(let num of nums) {
        if(num === 0) {
            zeroCount++
            continue
        }
        if(set.has(num)) {
            return false // 有对子，不是顺子
        }
        set.add(num)
        min = Math.min(min, num)
        max = Math.max(max, num)
    }
    return  max - min < 5 && zeroCount + set.size === 5;
}

// 78、圆圈中最后剩下的数字，约瑟夫问题
// 0,1,・・・,n-1 这 n 个数字排成一个圆圈，从数字 0 开始，每次从这个圆圈里删除第 m 个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。
// 例如，0、1、2、3、4 这 5 个数字组成一个圆圈，从数字 0 开始每次删除第 3 个数字，则删除的前 4 个数字依次是 2、0、4、1，最后剩下的数字是 3。
// 思路1: 构建环形链表，去遍历删除
function lastNumber(n, m) {
    let head = new ListNode(0)
    let prev = head
    // 构建环形链表
    for(let i = 1; i < n; i++) {
        let node = new ListNode(i)
        prev.next = node
        prev = prev.next
    }
    prev.next = head
    let cur = head

    while(n > 1) {
        // 移动到第m - 1个节点
        for(let i = 1; i < m - 1; i++) {
            cur = cur.next
        }
        // 删除 cur.next 节点
        cur.next = cur.next.next
        // cur 指向下一个节点
        cur = cur.next
        n--
    }
    return cur.val
}
// 思路2: 数学公式，约瑟夫问题的递推公式是f(n, m) = (f(n-1, m) + m) % n， 
function lastNumber(n, m) {
    let last = 0; // 基本情况：当只有一个人时，返回位置 0
    for (let i = 2; i <= n; i++) {
        last = (last + m) % i;
    }
    return last;
}

// 79、股票的最大利润
// 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
// 例如，一个数组为 [7,1,5,3,6,4]，在价格为 1 时买入，价格为 6 时卖出，最大利润就是 5。
// 思路1： 暴力法，时间复杂度o(n2),不推荐
// 思路2: 贪婪算法
function maxProfit(nums) {
    let minPrice = Infinity
    let maxProfit = 0
    for(let num of nums) {
        if(num < minPrice) {
            minPrice = num
        } else if(num - minPrice > maxProfit) {
            maxProfit = num - minPrice
        }
    }
    return maxProfit
}

// 80、求1 + 2 + 3...+ n
// 求 1+2+3+…+n，要求不能使用乘除法、for、while、if、else、switch、case 等关键字及条件判断语句（A?B:C）
// 思路1:用逻辑运算符 && 的短路特性来终止递归。
//      当 n > 0 时，n && (sum += sumNums(n - 1)) 会继续递归，当 n == 0 时，n 为假，递归终止
function sumN(n) {
    let sum = 0
    n && (sum += sumN(n - 1) + n)
    return sum
}
// 思路2: 利用类的构造函数，在构造函数中进行累加操作。通过创建 n 个类实例，每个实例的构造函数中进行累加。
//      可以利用类的静态变量存储累加结果
class Sum {
    static res = 0;
    constructor(n) {
        if (n > 0) {
            Sum.res += n;
            new Sum(n - 1);
        }
    }
}
function sumN(n) {
    Sum.res = 0;
    new Sum(n);
    return Sum.res;
}

// 81、不用加减乘除做加法
// 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号
// 思路位运算： 
//      两个整数在不考虑进位的情况下进行异或运算之后就是他们相加的结果，如1 ^ 2 = 3
//      在考虑进位的情况下，如1 ^ 3 = 2, 那么就需要来记录进位的情况了，是否进位可以用(a & b) << 1来算出应该要进到哪一位
function add(a,b) {
    while(b !== 0) {
       // 计算进位
       let carry = (a & b) << 1;

       // 计算不带进位的部分（即按位加法）
       a = a ^ b;

       // 进位需要加到下一次
       b = carry;
    }
    return a
}

// 82、交换两个数的值
function swap(a,b) {
    a = a + b
    b = a - b
    a = a - b
    return { a, b}
}

function swap(a,b) {
    a = a ^ b
    b = a ^ b
    a = a ^ b
    return { a, b }
}

// 83、构建乘积数组
// 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了 A[i] 之外的所有元素的乘积。不能使用除法。
// 例如，对于数组 A = [1,2,3,4]，B = [24,12,8,6]
// 思路1: 左右乘积，对于B[i], 可以分别计算left[i]和right[i],最后再计算B[i] = left[i] + right[i]
function structArr(A) {
    let len = A.length
    let left = new Array(len).fill(1)
    let right = new Array(len).fill(1)

    // 从左到右计算left[i]
    // left[0] 被赋值为 1，因为 A[0] 左边没有元素。
    // 循环从第二个开始
    for(let i = 1; i < len; i++) {
        left[i] = left[i - 1] * A[i - 1]
    }

    // 从右到左计算right[i]
    // right[n-1] 被赋值为 1，因为 A[n-1] 右边没有元素
    // 从倒数第二个开始
    for(let i = len - 2; i >= 0; i--) {
        right[i] = right[i + 1] * A[i + 1]
    }

    let B = []
    //计算B[i]
    for(let i = 0; i < len; i++) {
        B[i] = left[i] * right[i]
    }
    return B
}
// 思路2:可以只用一个辅助数组和一个变量，通过两次遍历完成。
// 第一次从左到右，将 left 存储在结果数组 B 中。
// 第二次从右到左，使用一个变量 right 存储右侧的累乘结果，并与 B[i] 相乘更新 B[i]。
function structArr(A) {
    let len = A.length
    let B = new Array(len).fill(1)

    // 从左到右计算left[i]
    for(let i = 1; i < len; i++) {
        B[i] = B[i - 1] * A[i - 1]
    }

    let right = 1
     // 从右到左计算right[i]
     for(let i = len - 1; i >= 0; i--) {
        B[i] = B[i] * right
        right = right * A[i]
     }

     return B
}

// 84、把字符串转换成整数
// 写一个函数 strToInt，将一个字符串转换成一个整数。函数需要满足以下要求：
//      首先，丢弃无用的开头空格字符，直到寻找到第一个非空格字符。
//      当第一个非空字符为正或者负号时，将该符号与之后尽可能多的连续数字组合起来，作为最终结果的正负号。
//      假如第一个非空字符是数字，则直接将其与之后尽可能多的连续数字字符组合起来。
//      字符串中有效的整数部分之后的多余字符将被忽略。
//      若字符串的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符，函数应返回 0。
//      假设环境只能存储 32 位有符号整数，其数值范围为[-2^31, 2^31 - 1] 。如果数值超过这个范围，返回相应的边界值。
// 思路： 步骤一：去除空格
//              使用 while 循环跳过开头的空格字符。
//       步骤二：判断正负号
//              检查第一个非空格字符是否为 + 或 -，并设置相应的正负标记。
//       步骤三：提取数字
//              遍历后续字符，将数字字符添加到结果中，同时考虑越界情况。
//       步骤四：越界处理
//              检查最终结果是否超出 32 位有符号整数的范围，超出则返回相应边界值。
function strToInt(str) {
    let i = 0; // 当前字符
    let sign = 1; // 正负号
    let res = 0;
    const max = Math.pow(2, 31) - 1;
    const min = -Math.pow(2, 31);

    // 跳过开头的空格字符
    while (i < str.length && str[i] === ' ') {
        i++;
    }

    // 判断正负号
    if (i < str.length && (str[i] === '+' || str[i] === '-')) {
        sign = str[i] === '+' ? 1 : -1;
        i++;
    }

    // 提取数字
    while (i < str.length && str[i] >= '0' && str[i] <= '9') {
        let digit = +str[i]; // 将字符转换为数字
        // 判断越界
        // max 的值为 2147483647，它是 32 位有符号整数的最大值，其十进制表示的最后一位数字是 7
        // 当 res 已经等于 Math.floor(max / 10) 时，即 res 已经达到了 214748364，此时需要考虑添加下一个数字 digit 是否会导致越界。
        // 如果 digit 大于 7，例如添加 8 或 9，那么结果将超过 2147483647，即超出了 32 位有符号整数的范围
        if (res > Math.floor(max / 10) || (res === Math.floor(max / 10) && digit > 7)) {
            return sign === 1 ? max : min;
        }
        res = res * 10 + digit;
        i++;
    }

    return res * sign;
}

// 85、二叉搜索树的最近公共祖先
// 给定一个二叉搜索树 (BST)，找到该树中两个指定节点的最近公共祖先 (LCA)。
// 最近公共祖先的定义为：
//      对于有根树 T 的两个节点 p 和 q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。
// 思路1: 递归，
//      利用二叉搜索树的性质（左子树节点值 < 根节点值 < 右子树节点值）。
//      如果 p 和 q 的值都小于当前节点的值，那么它们的最近公共祖先在左子树。
//      如果 p 和 q 的值都大于当前节点的值，那么它们的最近公共祖先在右子树。
//      否则，当前节点就是它们的最近公共祖先
function lowestCommonAncestor(root, p, q) {
    if(p.val < root.val && q.val < root.val) {
        return lowestCommonAncestor(root.left, p, q)
    } else if(p.val > root.val && q.val > root.val) {
        return lowestCommonAncestor(root.right, p, q)
    } else {
        return root
    }
}
// 思路2: 迭代法
//      从根节点开始迭代，根据 p 和 q 的值与当前节点值的大小关系，向下移动节点。
//      当 p 和 q 的值分别在当前节点值的两侧，或者其中一个等于当前节点值时，当前节点就是最近公共祖先
function lowestCommonAncestor(root, p, q) {
    let node = root
    while(node) {
        if(p.val < node.val && q.val < node.val) {
            node = node.left
        } else if (p.val > node.val && q.val > node.val) {
            node = node.right
        } else {
            return node
        }
    }
}

// 86、整数除法
// 输入 2 个int型整数，它们进行除法计算并返回商，要求不得使用乘号*、除号/及求余符号%。当发生溢出时，返回最大的整数值。假设除数不为 0。
// 例如，输入 15 和 2，输出 15/2 的结果，即 7。
// 思路： 用减法实现，被除数不断减去除数，最后得出答案，但这样的太慢了，可以采用策略加快节奏
//      a、如果当被除数大于除数时，先比较判断被除数是否大于除数的 2 倍、4 倍、8 倍等。若被除数最多大于除数的2的k次方倍，
//          就将被除数减去除数的2的k次方倍，然后对剩余的被除数重复此步骤。这样每次将除数翻倍，优化后的时间复杂度是o(logn)
//      b、对于有负数的情况，先将正数都转换成负数，用优化后的减法计算两个负数的除法，再根据需要调整商的正负号。
//          因为将任意正数转换为负数不会溢出，而将最小的整数-2147483648转换为正数会溢出。
//      c、整数除法只有一种情况会导致溢出，即(-2147483648) / (-1)，因为最大的正数为2147483647，2147483648超出了正数范围。
function divide(dividend, divsor) {
    const max = Math.pow(2, 31) - 1
    const min = Math.pow(-2, 31)
    // 结果溢出
    if(min === dividend && divsor === -1) {
        return max
    } 
    let resNegative = 2 // 记录符号变化的次数，初始是2，用来表示结果是否是正数


    // 把除数和被除数为负数的数变为正数，同时resNegative--，记录符号变化的次数，如果resNegative=1，那么代表结果是负数
    if(dividend < 0) {
        dividend = -dividend
        resNegative--
    }

    if(divsor < 0) {
        divsor = -divsor
        resNegative--
    }

    const divdeCore = (dividend, divsor) => {
        let res = 0
        // 当被除数大于除数时
        while(dividend > divsor) {

            let value = divsor
            // 初始商为 1
            let quotient = 1;
            // 被除数是否大于除数的 2 倍、4 倍、8 倍等
            while(dividend >= value + value) {
                quotient += quotient
                value += value
            }
            res += quotient
            dividend -= value
        }
        return res
    }

    let res = divdeCore(dividend, divsor)

    return resNegative === 1? -res : res
}

// 87、⼆进制加法
// 输入两个表示二进制的字符串，输出它们相加的结果，结果也用二进制字符串表示。
// 例如，输入 "11" 和 "1"，输出 "100"；输入 "1010" 和 "1011"，输出 "10101"
// 思路： 如果把字符串转成十进制数字相加再转成二进制是可以的，但如果字符串长度过长，可能会造成数字溢出，所以还是得从字符串的角度来相加，从最低位开始相加
function addBinary(a,b) {
    const maxLen = Math.max(a.length, b.length);
    // 为短的字符串在前面补0
    a = a.padStart(maxLen, '0');
    b = b.padStart(maxLen, '0');

    let takeOver = 0; // 是否进位
    let res = ''
    for(let i = maxLen - 1; i >= 0; i--) {
        let sum = parseInt(a[i]) + parseInt(b[i]) + takeOver
        if(sum >= 2) { // 需要进位
            takeOver = 1
            res =  (sum - 2) + res
        } else {
            takeOver = 0
            res = sum + res
        }
    }

    if(takeOver) {
        res = 1 + res
    }
    return res
}

// 88、前 n 个数字二进制中 1 的个数
// 输入一个非负整数 n，请计算 0 到 n 中每个数字的二进制表示中 1 的个数，并将结果作为一个数组返回。
// 例如，输入 n = 2，返回 [0,1,1]，因为 0 的二进制表示是 0（1 的个数为 0），1 的二进制表示是 1（1 的个数为 1），2 的二进制表示是 10（1 的个数为 1）
// 思路1: 位运算消1，统计一个数中1的个数(num - 1) & num
function getNumberOf1(n) {
    const getNumber1 = (num) => {
        let count = 0
        while(num) {
            num = (num - 1) & num
            count++
        }
        return count
    }

    let res = []
    for(let i = 0; i <= n; i++) {
        res.push(getNumber1(i))
    }
    return res
}
// 思路2：观察规律可以发现，对于数字 i，如果 i 是偶数，那么 i 的二进制表示中 1 的个数和 i / 2 的二进制表示中 1 的个数相同；
//      如果 i 是奇数，那么 i 的二进制表示中 1 的个数比 i - 1 的二进制表示中 1 的个数多 1。
function getNumberOf1(n) {
    let res = [0]
    for(let i = 1; i <= n; i++) {
        if(i % 2 == 0) {
            res.push(res[i >> 1])
        } else {
            res.push(res[i - 1] + 1)
        }
    }
    return res
}

// 89、只出现一次的数字，跟第70题是同一个

// 90、单词⻓度的最⼤乘积
// 输入一个字符串数组words，请计算不包含相同字符的两个字符串words[i]和words[j]的长度乘积的最大值。
// 如果所有字符串都包含至少一个相同字符，那么返回 0。假设字符串中只包含英文字母小写字符。
// 例如，输入的字符串数组words为["abcw","foo","bar","fxyz","abcdef"]，数组中的字符串bar与foo没有相同的字符，它们长度的乘积为 9。
// abcw与fxyz也没有相同的字符，它们长度的乘积为 16，这是该数组不包含相同字符的一对字符串的长度乘积的最大值。
// 思路1: 哈希表记录
//      a、对于每个字符串，创建一个长度为 26 的布尔型数组（模拟哈希表）来记录字符串中每个英文字母是否出现。
//      b、遍历字符串数组，对于每个字符串，将其中出现的字母在对应的布尔型数组中标记为true。
//      c、然后通过两层循环遍历字符串数组，对于每一对字符串，检查它们对应的布尔型数组中是否有相同位置都为true的情况，如果没有，则计算它们长度的乘积，并更新最大值。
function maxProduct(words) {
    // 用来记录每个字符串中，26个字母是否出现
    let flag = new Array(words.length).fill(0).map(() => new Array(26).fill(false))

    // 遍历字符串数组，对于每个字符串，将其中出现的字母在对应的布尔型数组中标记为true。
    for(let i = 0; i < words.length; i++) {
        for(let char of words[i]) {
            flag[i][char.charCodeAt() - 'a'.charCodeAt()] = true
        }
    }
    let res = 0
    // 两层循环遍历字符串数组
    for(let i = 0; i < words.length; i++) {
        for(let j = i + 1; j < words.length; j++) {
            let k = 0 // a-z中的第一个字符a开始
            while(k < 26) {
                if(flag[i][k] && flag[j][k]) {
                    break
                }
                k++
            }
            // k等于26，代表没有重复的
            if(k === 26) {
                res = Math.max(res, words[i].length * words[j].length)
            }
        }
    }
    return res
}
// 思路2: 在思路1的基础上使用位运算来降低时间复杂度和内存占用
//      使用二进制数来表示字符串中每个字母是否出现，如'abc'可以用00000000000000000000000111表示，'de'可以用00000000000000000001100000表示
//      再检查两个字符串是否有重复的字符时可以使用前面的二进制记录相与，如00000000000000000000000111 & 00000000000000000001100000 = 0，代表没有重复字符
function maxProduct(words) {
    let flag = new Array(words.length).fill(0)
    for(let i = 0; i < words.length; i++) {
        for(let char of words[i]) {
            // 把对应字母的位标记为 1
            // 1 << (char.charCodeAt() - 'a'.charCodeAt()),例如 1 << 2 -> 00000000000000000000000100
            // flag[i] | 1 << 2,例如 00000000000000000000000001 | 00000000000000000000000100 = 00000000000000000000000101
            flag[i] =  flag[i] | (1 << (char.charCodeAt() - 'a'.charCodeAt()))
        }
    }

    let res = 0
    for(let i = 0; i < words.length; i++) {
        for(let j = i; j < words.length; j++) {
            if((flag[i] & flag[j]) === 0) { // 没有重复数据
                res = Math.max(res, words[i].length * words[j].length)
            }
        }
    }
    return res
}

// 91、排序数组中的两个数字之和
// 输入一个递增排序的数组和一个值k，请问如何在数组中找出两个和为k的数字并返回它们的下标？假设数组中存在且只存在一对符合条件的数字，同时一个数字不能使用两次。
// 例如，输入数组[1, 2, 4, 6, 10]，k的值为 8，数组中的数字 2 与 6 的和为 8，它们的下标分别为 1 与 3。
// 思路1: 暴力法，两个for循环
// 思路2: 哈希表存储所有数字，然后再遍历一遍数字i，同时查询哈希表里有没有存在 k - i， 时间复杂度o(n), 空间复杂度o(n)
// 思路3: 二分查找，遍历一遍数字i，同时使用二分查找查找k - i，时间复杂度是o(nlogn)
// 思路4: 双指针，p1指向首位， p2指向末尾,时间复杂度o(n),空间复杂度o(1)
function twoSum(numbers, target) {
    let p1 = 0
    let p2 = numbers.length - 1
    while(p1 < p2) {
        if(numbers[p1] + numbers[p2] > target) {
            p2--
        } else if(numbers[p1] + numbers[p2] < target) {
            p1++
        } else {
            return [numbers[p1], numbers[p2]]
        }
    }
    return []
}

// 92、数组中和为0的3个数字
// 输入一个数组，如何找出数组中所有和为 0 的 3 个数字的三元组？需要注意的是，返回值中不得包含重复的三元组。
// 例如，在数组[-1, 0, 1, 2, -1, -4]中有两个三元组的和为 0，它们分别是[-1, 0, 1]和[-1, -1, 2]
// 思路1: 暴力法，3个循环，时间复杂度是o(n2)
// 思路2: 双指针，先排序，然后使用一个循环，确定第一个数字，然后再使用两个指针分别指向下一个和最后一个
//      注意的是可能有重复的结果，所以注意跳过重复的数字
function threeSum(numbers) {
    if(numbers.length < 3) {
        return []
    }
    numbers.sort((a, b) => a - b);

    let res = []
    // 注意遍历到numbers.length - 2，因为最后两个数字无法形成3元组
    for(let i = 0; i < numbers.length - 2; i++) {
        // 跳过重复的数字,即跳过重复的结果
        if (i > 0 && numbers[i] === numbers[i - 1]) {
            continue;
        }

        // 提前剪枝：当前数字为正数时，不可能有满足条件的三元组
        if (numbers[i] > 0) break;

        let p1 = i + 1
        let p2 = numbers.length - 1
        while(p1 < p2) {
            let sum = numbers[p1] + numbers[p2] + numbers[i]
            if(sum > 0) {
                p2--
            } else if(sum < 0) {
                p1++
            } else {
                res.push([numbers[i], numbers[p1], numbers[p2]])
                // 跳过重复的p1，p2
                while(p1 < p2 && numbers[p1] === numbers[p1 + 1]) {
                    p1++
                }

                while(p1 < p2 && numbers[p2] === numbers[p2 - 1]) {
                    p2--
                }
                // 左右指针继续收缩
                p1++
                p2--
            }
        }
    }
    return res
}

// 93、和大于等于k的最短子数组
// 输入一个正整数组成的数组和一个正整数k，请问数组中和大于或等于k的连续子数组的最短长度是多少？如果不存在所有数字之和大于或等于k的子数组，则返回 0。
// 例如，输入数组[5, 1, 4, 3]，k的值为 7，和大于或等于 7 的最短连续子数组是[4, 3]，因此输出它的长度 2。
// 思路： 双指针
//      a、使用两个指针left和right来表示子数组的范围，初始时都指向数组的第一个元素。
//      b、用变量sum来记录当前子数组的和。
//      c、不断移动right指针，扩大子数组的范围，同时更新sum的值，直到sum大于或等于k。
//      d、当sum大于或等于k时，开始移动left指针，缩小子数组的范围，同时更新sum的值，在缩小范围的过程中，记录子数组的最短长度。
//      重复上述步骤，直到right指针到达数组的末尾。
function minSubArrayLen(numbers, k) {
    let sum = 0
    let left = 0
    let minLen = Infinity
    for (let right = 0; right < numbers.length; right++) {
        sum += numbers[right] // 右指针扩展窗口值大小

        // 收缩窗口直到 sum < k
        while(sum >= k) {
            minLen = Math.min(minLen, right - left + 1)
            sum -= numbers[left]
            left++
        }
    }
    return minLen === Infinity ? 0 : minLen
}

// 94、乘积小于k的子数组
// 输入一个由正整数组成的数组和一个正整数k，请问数组中有多少个数字乘积小于k的连续子数组？
// 例如，输入数组[10, 5, 2, 6]，k的值为 100，有 8 个子数组的所有数字的乘积小于 100，它们分别是[10]、[5]、[2]、[6]、[10, 5]、[5, 2]、[2, 6]和[5, 2, 6]
// 思路：跟上面的方法类似，滑动窗口
//      a、初始化两个指针left和right，都指向数组的第一个元素。
//      b、使用变量product来记录当前子数组的乘积。
//      c、不断移动right指针，扩大子数组的范围，同时更新product的值，计算当前子数组的乘积。
//      d、当product大于或等于k时，开始移动left指针，缩小子数组的范围，同时更新product的值，在缩小范围的过程中，计算满足条件的子数组数量。
//      重复上述步骤，直到right指针到达数组的末尾。
function numSubarrayProductLessThanK(numbers, k) {
    if (k <= 1) return 0; // 如果 k <= 1，直接返回 0，因为没有子数组能满足条件

    let left = 0
    let count = 0
    let product = 1
    for(let right = 0; right < numbers.length; right++) {
        product *= numbers[right]

        // 收缩窗口
        while(product >= k) {
            product /= numbers[left]
            left++
        }
        
        // 当前窗口 [left, right] 内的所有子数组
        count += right - left + 1;
    }

    return count
}

// 95、和为k的子数组
// 输入一个整数数组和一个整数 k，请问数组中有多少个数字之和等于 k 的连续子数组？
// 例如，输入数组 [1, 1, 1]，k 的值为 2，有 2 个连续子数组之和等于 2
// 思路： 数组里有如果包含负数，且强调连续性，使用双指针就无法保证和的变化是单调的，所以这个题没法使用双指针
//    可以先计算从数组下标为 0 开始到以每个数字为结尾的子数组之和，然后通过哈希表来统计满足和为 k 的子数组个数。
//    具体来说，在从头到尾逐个扫描数组中的数字时求出前 i 个数字之和 sum，并将其保存下来。
//    同时，对于每个 sum，需要知道在 i 之前存在多少个 j 并且前 j 个数字之和等于 sum - k，因为这样的话从第 j + 1 个数字开始到第 i 个数字结束的子数组之和就是 k。
//    所以，对每个 i，不但要保存前 i 个数字之和，还要保存每个和出现的次数

function findSubarraysWithSum(numbers, k) {
    // 存储和及其出现的次数
    let sumMap = new Map()
    sumMap.set(0, 1) // // 初始化，表示和为0的子数组出现了一次（空数组）
    let sum = 0
    let count = 0
    for(let i = 0; i < numbers.length; i++) {
        sum += numbers[i]

         // 检查 值为sum - k的前缀和是否已经出现过
         // 出现过的话就代表有前缀和sum[j] + k = sum[i], j + 1到i之间的子数组元素之和是k
        let diff = sum - k
        if(sumMap.has(diff)) {
            count += sumMap.get(diff)
        } 

        // 更新 sumMap，记录当前 前缀和sum 的出现次数
        if(sumMap.has(sum)) {
            sumMap.set(sum, sumMap.get(sum) + 1)
        } else {
            sumMap.set(sum, 1)
        }
    }

    return count
}

// 96、和为 0 的最长子数组
// 输入一个整数数组，求其和为 0 的最长子数组的长度。
// 例如，在数组 [1, -1, 3, -2, 2] 中，和为 0 的最长子数组是 [1, -1]，长度为 2。
// 思路： 
//      a、利用前缀和的性质，如果在数组的两个不同位置 i 和 j（i < j），前缀和相等，则它们之间的子数组的和为 0。
//          假设 sum[i] 是从数组开头到索引 i 的前缀和，如果有 sum[j] == sum[i]（j > i），那么从 i+1 到 j 的子数组和为 0
//      b、使用哈希表记录每个前缀和第一次出现的索引位置。每次计算新的前缀和时，检查是否在哈希表中已存在：
//          如果存在，计算两个索引之间的距离，更新最长长度。如果不存在，将当前前缀和及其索引存入哈希表。
//      c、前缀和为 0 时，说明从数组起始到当前索引的子数组的和为 0，此时需要更新最长长度。
function findMaxLen(numbers) {
    let sumMap = new Map()
    let sum = 0
    let maxLen = 0

    sumMap.set(0, -1)
    for(let i = 0; i < numbers.length; i++) {
        sum += numbers[i]
        // 如果有 sum[j] == sum[i]（j > i），那么从 i+1 到 j 的子数组和为 0
        if(sumMap.has(sum)) {
            const prevIndex = sumMap.get(num)
            maxLen = Math.max(maxLen, i - prevIndex)
        } else {
            sumMap.set(sum, i)
        }
    }
    return maxLen
}

// 97、0和1个数相同的子数组
// 输入一个只包含0和1的数组，求0和1的个数相同的最长连续子数组的长度。
// 例如：在数组 [0, 1, 0] 中，有两个子数组 [0, 1] 和 [1, 0] 包含相同个数的0和1，它们的长度都是2，因此输出 2。
// 思路： 将0变成-1，那么这个题就变成和为0的最长子数组这道题了， 时间复杂度o(n),空间复杂度o(n)
// 总结： 前缀和与哈希表 的结合是解决连续子数组问题的核心
function findMaxLen(numbers) {
    let sumMap = new Map()
    let sum = 0
    let maxLen = 0
    for(let i = 0; i < numbers.length; i++) {
        sum += (numbers[i] === 0 ? -1 : 1)
        if(sumMap.has(sum)) {
            let prevIndex = sumMap.get(sum)
            maxLen = Math.max(maxLen, i - prevIndex)
        } else {
            sumMap.set(sum, i)
        }
    }

    return maxLen
}

// 98、左右两边子数组的和相等
// 输入一个整数数组，找到一个数字使得它左边的子数组和等于右边的子数组和。返回该数字的索引。
// 如果有多个这样的数字，返回最左边的数字的索引；如果不存在这样的数字，返回 -1
// 思路：
//      计算数组中所有数字的总和 totalSum。
//    	遍历数组，逐步计算当前元素左边的子数组和 leftSum。
//      对于当前元素，右边的子数组和可以通过公式计算rightSum = totalSum - leftSum - numbers[i]
function pivotIndex(numbers) {
    let totalSum = numbers.reduce((acc, num) => acc + num, 0)
    let leftSum = 0
    for(let i = 0; i < numbers.length; i++) {
        const rightSum = totalSum - leftSum - numbers[i]
        if(leftSum === rightSum) {
            return i
        } else {
            leftSum += numbers[i]
        }
    }
    return -1
}

// 99、调整数组使奇数位于偶数前面
// 输入一个整数数组，请调整数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。要求在调整之后，奇数和奇数、偶数和偶数之间的相对位置不变。
// 例如：
// •	输入 [1, 2, 3, 4]，输出 [1, 3, 2, 4]。
// •	输入 [2, 4, 6]，输出 [2, 4, 6]（没有奇数）。
// •	输入 [1, 3, 5]，输出 [1, 3, 5]（没有偶数）。
// 思路： 双指针
function reorderArray(numbers) {
    if (!numbers || numbers.length === 0) return [];

    let len = numbers.length;
    let left = 0;
    let right = len - 1;

    while (left < right) {
        let isLeftOdd = (numbers[left] & 1) === 1;
        let isRightEven = (numbers[right] & 1) === 0;

        if (isLeftOdd) {
            left++;
        } else if (isRightEven) {
            right--;
        } else {
            [numbers[left], numbers[right]] = [numbers[right], numbers[left]];
            left++;
            right--;
        }
    }

    return numbers;
}

// 100、二维子矩阵的数字之和
//  输入一个二维矩阵，如何计算给定左上角坐标和右下角坐标的子矩阵的数字之和？对于同一个二维矩阵，计算子矩阵的数字之和的函数可能由于输入不同的坐标而被反复调用多次
// 例如：输入下面的二维矩阵，输入左上角坐标(2,1)和右下角坐标(4,3)，输出8
// [ 
//    [3, 0, 1, 4, 2],
//    [5, 6, 3, 2 ,1],
//    [1, 2, 0, 1, 5],
//    [4, 1, 0, 1, 7],
//    [1, 0, 3, 0, 5]
//]
// 思路： 和数组前缀和类似，求二维数组的前缀和
function subMatrixSum(matrix, row1, col1, row2, col2) {
    if(!matrix || matrix.length === 0 || matrix[0].length === 0) {
        return null
    }

    let rows = matrix.length
    let columns  = matrix[0].length
    // 存储二维矩阵的前缀和
    let prefix = new Array(rows).fill(0).map(() => new Array(columns).fill(0))

    // 计算二维矩阵前缀和
    for(let i = 0; i < rows; i++) {
        for(let j = 0; j < columns; j++) {
            prefix[i][j] = matrix[i][j] 
                + (i > 0 ? prefix[i - 1][j]  : 0) 
                + (j > 0 ? prefix[i][j - 1] : 0) 
                - (i > 0 && j > 0 ? prefix[i - 1][j - 1] : 0)
                
        }
    }

    // 计算子矩阵的和
    let res = prefix[row2][col2]
    if(row1 > 0) {
        res -= prefix[row1 - 1][col2]
    }
    if(col1 > 0) {
        res -= prefix[row2][col1 - 1]
    }

    if(row1 > 0 && col1 > 0) {
        res += prefix[row1 - 1][col1 - 1]
    }
    return res
}

// 101、字符串中的变位词
// 输入字符串 s1 和 s2，判断字符串 s2 中是否包含字符串 s1 的某个变位词。假设两个字符串中只包含英文字母小写。
// 例如，字符串 s1 为 "ac"，字符串 s2 为 "dgcaf"，由于字符串 s2 中包含字符串 s1 的变位词 "ca"，因此输出为 true。
//      如果字符串 s1 为 "ab"，字符串 s2 为 "dgcaf"，则输出为 false
// 思路：
function checkInclusion(s1, s2) {
    let len1 = s1.length
    let len2 = s2.length

    if(len1 > len2) {
        return false
    }

   
}