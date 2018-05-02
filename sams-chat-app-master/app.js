class ChatAppController {
    constructor(config) {
        this.config = config;
        this.session_id = Date.now() / 1000000000;
        this.feedbackGiven = true;
    }

    // This initialises the chat app by creating the required interface
    init() {
        console.log("init()::");
        this.container = document.getElementById('chat-app');
        this.container.classList.add('chat-app-container');
        this.container.classList.add(this.config.theme);
        let header = document.createElement('div');
        let appName = document.createElement('label');
        appName.innerText = this.config.appName;
        header.appendChild(appName);
        header.addEventListener("click", this.dockWindow.bind(this));
        // var minimise = document.createElement('span');
        // minimise.innerText = "-";
        // minimise.className = "dock-icon";
        // header.appendChild(minimise);
        header.classList = "header";
        this.container.appendChild(header);

        this.chatMessageContainer = document.createElement('div');
        this.chatMessageContainer.className = "chat-message-container";
        this.loadingIcon = document.createElement('div');
        this.loadingIcon.className = "loading";
        this.container.appendChild(this.chatMessageContainer);

        let inputContainer = document.createElement('div');
        inputContainer.className = "input-wrapper";
        this.inputTag = document.createElement('input');
        this.inputTag.className = "user-input";
        this.inputTag.placeholder= "Enter message..."
        this.inputTag.addEventListener('keyup', e => {
            if (e.keyCode == 13) {
                this.onNewQuestion();
            }
        });
        inputContainer.appendChild(this.inputTag);
        let enterBtn = document.createElement('button');
        enterBtn.className = "enter-btn";
        enterBtn.innerText = "Ask";
        enterBtn.addEventListener("click", this.onNewQuestion.bind(this));
        inputContainer.appendChild(enterBtn);
        this.container.appendChild(inputContainer);
        this.addMessage("Hey! \nI am your Legal assistant. How can I help you today? \n\nType in your question to progress.", false);
        this.feedbackContainer = document.createElement('div');
        let thumbsUp = document.createElement('span');
        thumbsUp.className = 'glyphicon glyphicon-thumbs-up';
        thumbsUp.addEventListener("click", () => {
            this.onFeedback(true);
        });
        let thumbsDown = document.createElement('span');
        thumbsDown.className = 'glyphicon glyphicon-thumbs-down';
        thumbsDown.addEventListener("click", () => {
            this.onFeedback(false);
        });
        this.feedbackContainer.appendChild(thumbsUp);
        this.feedbackContainer.appendChild(thumbsDown);

        // this.addMessage("After the order is placed, you will receive an email confirming your purchase. After the order ships, you will receive a second email which will provide tracking information. You can also find this online on the item detail page under your Order History", false);
    }

    onFeedback(isUpVote) {
        this.feedbackGiven = true;
        this.feedbackContainer.remove();
        isUpVote = isUpVote ? 1 : 0;
        fetch(`${this.config.serverUrl}/feedback/insert`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "question": this.question, "input_question": this.input_question, "state": this.state, "intent": this.intent, "summ_answer": this.summ_answer, "detail_answer": this.detail_answer, "thumbs_up": isUpVote })
        })
            .then(res => res.json())
            .then(data => console.info("Success::", data))
            .catch(error => {
                console.error("Failed to get feedback response", error);
            });
    }

    // Adds the entered question or answer to the list
    addMessage(data, self) {
        let container = document.createElement('div');
        let msgWrapper = document.createElement('div');
        let msgContentWrapper = document.createElement('div');
        msgContentWrapper.className = "message-wrapper"
        let msg = document.createElement('span');
        msg.className = "message-content";
        msg.innerText = data;
        let iconNode = document.createElement('span');
        if(self) {
            container.className = "self-msg-wrapper";
            msgWrapper.className = "message self";
            iconNode.className = "icon self-icon";
        } else {
            container.className = "bot-msg-wrapper";
            msgWrapper.className = "message bot";
            iconNode.className = "icon bot-icon";
        }
        msgContentWrapper.appendChild(iconNode);
        msgContentWrapper.appendChild(msg);
        let timeStamp = document.createElement('div');
        let now = new Date();
        timeStamp.className = "time-stamp";
        let hrs = now.getHours();
        if(hrs < 10) {
            hrs = '0' + hrs;
        }
        let mins = now.getMinutes();
        if(mins < 10) {
            mins = '0' + mins;
        }
        timeStamp.innerText = hrs + ":" + mins;
        msgWrapper.appendChild(msgContentWrapper);
        container.appendChild(msgWrapper);
        container.appendChild(timeStamp);
        this.chatMessageContainer.appendChild(container);
        setTimeout(() => this.chatMessageContainer.scrollTop = this.chatMessageContainer.scrollHeight, 0);
    }

    // Makes a call to the server to get the answer for the entered question
    getAnswer(question) {
        var _this = this;
        this.question = question;
        this.feedbackGiven = false;
        fetch(`${this.config.serverUrl}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "query": question, "session_id": this.session_id })
        })
            .then(res => res.json())
            .then(data => {
                console.info("Success::", data);
                this.addMessage(data.reply, false);
                this.input_question = data.input_question;
                this.intent = data.intent;
                this.summ_answer = data.summ_answer;
                this.state = data.state;
                this.detail_answer = data.detail_answer;
                this.chatMessageContainer.appendChild(this.feedbackContainer);
                this.loadingIcon.remove();
            })
            .catch(error => {
                console.error("Failed to get response", error);
                this.loadingIcon.remove();
            });
    }

    // Once the question has been entered and clicked enter button
    onNewQuestion() {
        console.log("onNewQuestion()::");
        if(this.inputTag.value !== '') {
            if (!this.feedbackGiven) {
                this.onFeedback(true);
            }
            this.addMessage(this.inputTag.value, true);
            this.getAnswer(this.inputTag.value);
            this.chatMessageContainer.appendChild(this.loadingIcon);
            this.inputTag.value = '';
        }
    }

    dockWindow() {
        if(this.container.classList.contains('docking')) {
            this.container.classList.remove('docking');
            this.container.classList.add('opening');
        } else {
            this.container.classList.remove('opening');
            this.container.classList.add('docking');
        }
    }
}
