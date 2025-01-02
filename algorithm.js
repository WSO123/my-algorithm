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
        this.stack1 = []
        this.stack2 = []
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
            if(number >= start && number <= end) {
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
 